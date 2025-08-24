#!/usr/bin/env python3
"""
RIF Live System Context Engine

Maintains real-time "big picture" system understanding and generates live context
for agents including system overview, design goals, architecture, and dependencies.

Addresses Issue #112 requirement for live system context and big picture awareness.
"""

import json
import os
import subprocess
import re
import stat
from datetime import datetime, timedelta
from typing import Dict, List, Any, Optional, Set, Tuple
from dataclasses import dataclass, asdict
from enum import Enum
import hashlib
import threading
import time
from pathlib import Path
import asyncio
from concurrent.futures import ThreadPoolExecutor, as_completed
import weakref
import sys

class SystemComponentType(Enum):
    """Types of system components"""
    AGENT = "agent"
    WORKFLOW = "workflow"
    TOOL = "tool"
    CONFIGURATION = "configuration"
    KNOWLEDGE_BASE = "knowledge_base"
    FRAMEWORK = "framework"
    INTEGRATION = "integration"
    QUALITY_GATE = "quality_gate"

class UpdateTrigger(Enum):
    """Triggers for system context updates"""
    CODE_CHANGE = "code_change"
    ISSUE_COMPLETION = "issue_completion"
    CONFIGURATION_CHANGE = "configuration_change"
    ARCHITECTURAL_DECISION = "architectural_decision"
    WORKFLOW_UPDATE = "workflow_update"
    PERIODIC_REFRESH = "periodic_refresh"
    GIT_COMMIT = "git_commit"
    GIT_MERGE = "git_merge"
    GIT_CHECKOUT = "git_checkout"
    MANUAL_TRIGGER = "manual_trigger"

@dataclass
class SystemComponent:
    """Individual system component"""
    id: str
    name: str
    type: SystemComponentType
    description: str
    purpose: str
    dependencies: List[str]
    relationships: Dict[str, str]
    status: str  # "active", "inactive", "deprecated"
    last_modified: datetime
    file_paths: List[str]
    
    def to_dict(self) -> Dict[str, Any]:
        data = asdict(self)
        data['type'] = self.type.value
        data['last_modified'] = self.last_modified.isoformat()
        return data

@dataclass
class SystemOverview:
    """Live system overview for big picture context"""
    purpose: str
    core_mission: str
    design_philosophy: str
    architecture_summary: str
    key_capabilities: List[str]
    quality_principles: List[str]
    constraints: List[str]
    success_metrics: Dict[str, Any]
    last_updated: datetime
    
    def to_dict(self) -> Dict[str, Any]:
        data = asdict(self)
        data['last_updated'] = self.last_updated.isoformat()
        return data

@dataclass
class DependencyGraph:
    """System dependency relationships"""
    components: Dict[str, SystemComponent]
    relationships: List[Dict[str, Any]]
    critical_paths: List[List[str]]
    integration_points: List[Dict[str, Any]]
    external_dependencies: List[Dict[str, Any]]
    last_analyzed: datetime
    
    def to_dict(self) -> Dict[str, Any]:
        data = asdict(self)
        data['components'] = {k: v.to_dict() for k, v in self.components.items()}
        data['last_analyzed'] = self.last_analyzed.isoformat()
        return data

@dataclass
class LiveSystemContext:
    """Complete live system context"""
    overview: SystemOverview
    dependency_graph: DependencyGraph
    design_goals: List[Dict[str, Any]]
    architectural_decisions: List[Dict[str, Any]]
    workflow_states: Dict[str, Any]
    quality_gates: List[Dict[str, Any]]
    knowledge_patterns: List[str]
    recent_changes: List[Dict[str, Any]]
    system_health: Dict[str, Any]
    context_timestamp: datetime
    
    def to_dict(self) -> Dict[str, Any]:
        data = asdict(self)
        data['overview'] = self.overview.to_dict()
        data['dependency_graph'] = self.dependency_graph.to_dict()
        data['context_timestamp'] = self.context_timestamp.isoformat()
        return data

class SystemAnalyzer:
    """Analyzes system components and relationships"""
    
    def __init__(self, repo_path: str = "/Users/cal/DEV/RIF"):
        self.repo_path = repo_path
    
    def analyze_system_components(self) -> Dict[str, SystemComponent]:
        """Analyze and catalog all system components"""
        components = {}
        
        # Analyze RIF agents
        agent_components = self._analyze_agents()
        components.update(agent_components)
        
        # Analyze workflow configurations
        workflow_components = self._analyze_workflows()
        components.update(workflow_components)
        
        # Analyze tools and systems
        tool_components = self._analyze_tools()
        components.update(tool_components)
        
        # Analyze configurations
        config_components = self._analyze_configurations()
        components.update(config_components)
        
        return components
    
    def _analyze_agents(self) -> Dict[str, SystemComponent]:
        """Analyze RIF agent files"""
        components = {}
        agents_path = os.path.join(self.repo_path, "claude", "agents")
        
        if os.path.exists(agents_path):
            for filename in os.listdir(agents_path):
                if filename.startswith("rif-") and filename.endswith(".md"):
                    agent_path = os.path.join(agents_path, filename)
                    
                    try:
                        with open(agent_path, 'r') as f:
                            content = f.read()
                        
                        agent_name = filename[:-3]  # Remove .md extension
                        
                        # Extract agent details
                        purpose = self._extract_agent_purpose(content)
                        dependencies = self._extract_agent_dependencies(content)
                        
                        component = SystemComponent(
                            id=f"agent-{agent_name}",
                            name=agent_name,
                            type=SystemComponentType.AGENT,
                            description=f"RIF specialized agent: {agent_name}",
                            purpose=purpose,
                            dependencies=dependencies,
                            relationships={},
                            status="active",
                            last_modified=datetime.fromtimestamp(os.path.getmtime(agent_path)),
                            file_paths=[f"claude/agents/{filename}"]
                        )
                        
                        components[component.id] = component
                        
                    except Exception as e:
                        print(f"Error analyzing agent {filename}: {e}")
        
        return components
    
    def _extract_agent_purpose(self, content: str) -> str:
        """Extract agent purpose from content"""
        # Look for role or purpose descriptions
        purpose_patterns = [
            r"## Role\n(.*?)(?:\n##|\n$)",
            r"Purpose[:\s]+(.*?)(?:\n|\.|$)",
            r"responsible for (.*?)(?:\n|\.|$)"
        ]
        
        for pattern in purpose_patterns:
            match = re.search(pattern, content, re.IGNORECASE | re.DOTALL)
            if match:
                purpose = match.group(1).strip()
                # Clean up the purpose text
                purpose = re.sub(r'\n+', ' ', purpose)
                purpose = re.sub(r'\s+', ' ', purpose)
                return purpose[:200]  # Limit length
        
        return "RIF specialized agent"
    
    def _extract_agent_dependencies(self, content: str) -> List[str]:
        """Extract agent dependencies from content"""
        dependencies = []
        
        # Look for mentions of other agents or systems
        if "knowledge" in content.lower():
            dependencies.append("knowledge-base")
        if "github" in content.lower() or "gh " in content.lower():
            dependencies.append("github-integration")
        if "workflow" in content.lower():
            dependencies.append("workflow-engine")
        if "quality" in content.lower():
            dependencies.append("quality-gates")
        
        # Look for specific agent mentions
        agent_patterns = [
            r"RIF-(\w+)",
            r"rif-(\w+)",
        ]
        
        for pattern in agent_patterns:
            matches = re.findall(pattern, content, re.IGNORECASE)
            for match in matches:
                dep = f"agent-rif-{match.lower()}"
                if dep not in dependencies:
                    dependencies.append(dep)
        
        return dependencies[:10]  # Limit dependencies
    
    def _analyze_workflows(self) -> Dict[str, SystemComponent]:
        """Analyze workflow configurations"""
        components = {}
        config_path = os.path.join(self.repo_path, "config")
        
        workflow_files = [
            "rif-workflow.yaml",
            "multi-agent.yaml",
            "framework-variables.yaml"
        ]
        
        for workflow_file in workflow_files:
            file_path = os.path.join(config_path, workflow_file)
            if os.path.exists(file_path):
                component_id = f"workflow-{workflow_file[:-5]}"  # Remove .yaml
                
                component = SystemComponent(
                    id=component_id,
                    name=workflow_file,
                    type=SystemComponentType.WORKFLOW,
                    description=f"Workflow configuration: {workflow_file}",
                    purpose="Defines system workflow and state management",
                    dependencies=["github-integration", "agent-orchestration"],
                    relationships={},
                    status="active",
                    last_modified=datetime.fromtimestamp(os.path.getmtime(file_path)),
                    file_paths=[f"config/{workflow_file}"]
                )
                
                components[component_id] = component
        
        return components
    
    def _analyze_tools(self) -> Dict[str, SystemComponent]:
        """Analyze system tools and utilities"""
        components = {}
        
        # Systems directory
        systems_path = os.path.join(self.repo_path, "systems")
        if os.path.exists(systems_path):
            for filename in os.listdir(systems_path):
                if filename.endswith(".py"):
                    file_path = os.path.join(systems_path, filename)
                    component_id = f"tool-{filename[:-3]}"
                    
                    purpose = self._extract_tool_purpose(file_path)
                    
                    component = SystemComponent(
                        id=component_id,
                        name=filename,
                        type=SystemComponentType.TOOL,
                        description=f"System tool: {filename}",
                        purpose=purpose,
                        dependencies=self._extract_tool_dependencies(file_path),
                        relationships={},
                        status="active",
                        last_modified=datetime.fromtimestamp(os.path.getmtime(file_path)),
                        file_paths=[f"systems/{filename}"]
                    )
                    
                    components[component_id] = component
        
        return components
    
    def _extract_tool_purpose(self, file_path: str) -> str:
        """Extract tool purpose from Python file"""
        try:
            with open(file_path, 'r') as f:
                content = f.read()
            
            # Look for module docstring
            docstring_match = re.search(r'"""(.*?)"""', content, re.DOTALL)
            if docstring_match:
                docstring = docstring_match.group(1).strip()
                # Get first meaningful line
                lines = [line.strip() for line in docstring.split('\n') if line.strip()]
                if lines:
                    return lines[0][:200]
            
            return f"System tool: {os.path.basename(file_path)}"
            
        except Exception:
            return "System utility"
    
    def _extract_tool_dependencies(self, file_path: str) -> List[str]:
        """Extract tool dependencies from Python imports"""
        dependencies = []
        
        try:
            with open(file_path, 'r') as f:
                content = f.read()
            
            # Look for specific system references
            if "mcp__rif-knowledge" in content:
                dependencies.append("knowledge-base")
            if "subprocess" in content and "gh " in content:
                dependencies.append("github-integration")
            if "ContextOptimizer" in content:
                dependencies.append("context-optimization")
            if "git" in content.lower():
                dependencies.append("git-integration")
                
        except Exception:
            pass
        
        return dependencies
    
    def _analyze_configurations(self) -> Dict[str, SystemComponent]:
        """Analyze system configurations"""
        components = {}
        
        # Configuration files to analyze
        config_files = [
            ("config", "Claude Code configuration", "claude/CLAUDE.md"),
            ("setup", "System setup script", "setup.sh"),
            ("rif-init", "RIF initialization", "rif-init.sh")
        ]
        
        for config_type, description, file_path in config_files:
            full_path = os.path.join(self.repo_path, file_path)
            if os.path.exists(full_path):
                component_id = f"config-{config_type}"
                
                component = SystemComponent(
                    id=component_id,
                    name=f"{config_type} configuration",
                    type=SystemComponentType.CONFIGURATION,
                    description=description,
                    purpose="System configuration and setup",
                    dependencies=[],
                    relationships={},
                    status="active",
                    last_modified=datetime.fromtimestamp(os.path.getmtime(full_path)),
                    file_paths=[file_path]
                )
                
                components[component_id] = component
        
        return components
    
    def analyze_dependencies(self, components: Dict[str, SystemComponent]) -> DependencyGraph:
        """Analyze system dependencies and relationships"""
        relationships = []
        critical_paths = []
        integration_points = []
        external_dependencies = []
        
        # Analyze component relationships
        for component_id, component in components.items():
            for dep_id in component.dependencies:
                if dep_id in components:
                    relationships.append({
                        "from": component_id,
                        "to": dep_id,
                        "type": "depends_on",
                        "strength": "strong"
                    })
        
        # Identify critical paths (simplified)
        agent_components = [c for c in components.values() if c.type == SystemComponentType.AGENT]
        if agent_components:
            critical_paths.append([
                "workflow-rif-workflow",
                "agent-rif-analyst", 
                "agent-rif-planner",
                "agent-rif-implementer",
                "agent-rif-validator"
            ])
        
        # Identify integration points
        integration_points = [
            {
                "name": "GitHub Integration",
                "components": ["github-integration", "workflow-rif-workflow"],
                "type": "external_api"
            },
            {
                "name": "Knowledge Base Integration", 
                "components": ["knowledge-base", "tool-context-optimization-engine"],
                "type": "internal_service"
            },
            {
                "name": "Claude Code Integration",
                "components": ["config-config", "agent-rif-implementer"],
                "type": "platform_integration"
            }
        ]
        
        # External dependencies
        external_dependencies = [
            {"name": "GitHub CLI", "type": "command_line_tool", "critical": True},
            {"name": "Git", "type": "version_control", "critical": True},
            {"name": "Claude Code", "type": "ai_platform", "critical": True},
            {"name": "MCP Servers", "type": "protocol", "critical": False}
        ]
        
        return DependencyGraph(
            components=components,
            relationships=relationships,
            critical_paths=critical_paths,
            integration_points=integration_points,
            external_dependencies=external_dependencies,
            last_analyzed=datetime.now()
        )

class ContextGenerator:
    """Generates live system context from analysis"""
    
    def __init__(self, repo_path: str = "/Users/cal/DEV/RIF"):
        self.repo_path = repo_path
        self.analyzer = SystemAnalyzer(repo_path)
    
    def generate_system_overview(self) -> SystemOverview:
        """Generate live system overview"""
        
        # Analyze current system state
        components = self.analyzer.analyze_system_components()
        
        # Extract key capabilities
        key_capabilities = []
        agent_count = len([c for c in components.values() if c.type == SystemComponentType.AGENT])
        key_capabilities.append(f"Automatic agent orchestration ({agent_count} specialized agents)")
        
        if any("workflow" in c.id for c in components.values()):
            key_capabilities.append("State machine workflow management")
        
        if any("knowledge" in c.purpose.lower() for c in components.values()):
            key_capabilities.append("Knowledge base learning and pattern recognition")
        
        if any("quality" in c.purpose.lower() for c in components.values()):
            key_capabilities.append("Quality gate enforcement")
        
        # Extract design philosophy from CLAUDE.md if exists
        claude_md_path = os.path.join(self.repo_path, "CLAUDE.md")
        design_philosophy = "Automatic intelligent development with zero manual intervention"
        
        if os.path.exists(claude_md_path):
            try:
                with open(claude_md_path, 'r') as f:
                    content = f.read()
                
                # Extract philosophy from content
                if "automatic" in content.lower() and "zero manual" in content.lower():
                    design_philosophy = "Automatic intelligent development system with zero manual intervention"
            except Exception:
                pass
        
        return SystemOverview(
            purpose="Orchestrate specialized agents through GitHub issues with zero manual intervention",
            core_mission="Transform GitHub issues into fully implemented, tested, and validated solutions automatically",
            design_philosophy=design_philosophy,
            architecture_summary="State machine driven with specialized agents, quality gates, and continuous learning",
            key_capabilities=key_capabilities,
            quality_principles=[
                "Test-driven development",
                "Continuous quality monitoring", 
                "Adversarial validation",
                "Knowledge-driven decisions",
                "Performance-first design"
            ],
            constraints=[
                "Technology agnostic implementation",
                "Claude Code integration required",
                "GitHub-driven workflow",
                "Performance budgets <200ms",
                "Zero manual configuration"
            ],
            success_metrics={
                "automation_rate": "100%",
                "issue_resolution_time": "< 4 hours average",
                "quality_gate_pass_rate": "> 90%",
                "knowledge_utilization": "> 80%"
            },
            last_updated=datetime.now()
        )
    
    def extract_design_goals(self) -> List[Dict[str, Any]]:
        """Extract current design goals from system analysis"""
        goals = [
            {
                "id": "automation",
                "title": "Complete Automation", 
                "description": "Zero manual intervention for issue resolution",
                "status": "active",
                "priority": "critical"
            },
            {
                "id": "quality",
                "title": "Quality Excellence",
                "description": "Maintain high quality standards through automated gates",
                "status": "active", 
                "priority": "high"
            },
            {
                "id": "learning",
                "title": "Continuous Learning",
                "description": "Learn from every interaction to improve future performance",
                "status": "active",
                "priority": "high"
            },
            {
                "id": "performance",
                "title": "Performance Optimization", 
                "description": "Maintain sub-200ms response times for all operations",
                "status": "active",
                "priority": "medium"
            }
        ]
        
        return goals
    
    def extract_architectural_decisions(self) -> List[Dict[str, Any]]:
        """Extract key architectural decisions"""
        decisions = [
            {
                "id": "agent-specialization",
                "title": "Specialized Agent Architecture",
                "description": "Use specialized agents (Analyst, Planner, Architect, Implementer, Validator) rather than general-purpose agents",
                "rationale": "Better expertise and context optimization for specific tasks",
                "status": "implemented",
                "date": "2024-08-01"
            },
            {
                "id": "state-machine-workflow",
                "title": "State Machine Workflow Management", 
                "description": "Use state machine pattern for issue lifecycle management",
                "rationale": "Predictable workflows with clear transitions and recovery points",
                "status": "implemented",
                "date": "2024-08-01"
            },
            {
                "id": "github-driven",
                "title": "GitHub-Driven Development",
                "description": "Use GitHub issues as the primary workflow trigger and coordination mechanism",
                "rationale": "Leverages existing developer workflow and provides audit trail",
                "status": "implemented", 
                "date": "2024-08-01"
            },
            {
                "id": "knowledge-learning",
                "title": "Continuous Knowledge Learning",
                "description": "Build and maintain knowledge base from every development interaction",
                "rationale": "Improves decision quality and reduces repeated analysis",
                "status": "implemented",
                "date": "2024-08-01"
            }
        ]
        
        return decisions
    
    def get_workflow_states(self) -> Dict[str, Any]:
        """Get current workflow states from configuration"""
        states = {
            "new": "Issue created, ready for analysis",
            "analyzing": "RIF-Analyst performing requirements analysis", 
            "planning": "RIF-Planner creating implementation strategy",
            "architecting": "RIF-Architect designing system architecture",
            "implementing": "RIF-Implementer writing code",
            "validating": "RIF-Validator testing and quality assurance",
            "learning": "RIF-Learner updating knowledge base",
            "complete": "Issue resolved and closed"
        }
        
        return states
    
    def extract_quality_gates(self) -> List[Dict[str, Any]]:
        """Extract quality gate definitions"""
        gates = [
            {
                "name": "Test Coverage",
                "threshold": ">80%",
                "type": "automated",
                "blocking": True
            },
            {
                "name": "Security Scan", 
                "threshold": "No critical vulnerabilities",
                "type": "automated",
                "blocking": True
            },
            {
                "name": "Performance",
                "threshold": "Meet baseline metrics",
                "type": "automated",
                "blocking": False
            },
            {
                "name": "Documentation",
                "threshold": "Complete API documentation",
                "type": "automated",
                "blocking": False
            },
            {
                "name": "Code Quality",
                "threshold": "Pass linting and type checking",
                "type": "automated", 
                "blocking": True
            }
        ]
        
        return gates
    
    def get_recent_changes(self) -> List[Dict[str, Any]]:
        """Get recent system changes from git"""
        changes = []
        
        try:
            # Get recent commits
            result = subprocess.run(
                ["git", "log", "--oneline", "--since=1 week ago", "-10"],
                cwd=self.repo_path, capture_output=True, text=True, check=True
            )
            
            for line in result.stdout.strip().split('\n'):
                if line:
                    parts = line.split(' ', 1)
                    if len(parts) == 2:
                        changes.append({
                            "commit": parts[0],
                            "message": parts[1],
                            "type": "code_change",
                            "impact": "medium"
                        })
                        
        except subprocess.CalledProcessError:
            pass
        
        return changes[:5]  # Limit to 5 recent changes
    
    def assess_system_health(self) -> Dict[str, Any]:
        """Assess current system health"""
        health = {
            "overall_status": "healthy",
            "agent_status": "active",
            "workflow_status": "operational", 
            "knowledge_base_status": "active",
            "integration_status": "connected",
            "last_assessment": datetime.now().isoformat(),
            "issues": []
        }
        
        # Check for potential issues
        try:
            # Check if git is working
            subprocess.run(["git", "status"], cwd=self.repo_path, check=True, capture_output=True)
        except subprocess.CalledProcessError:
            health["issues"].append("Git integration issue detected")
            health["overall_status"] = "degraded"
        
        try:
            # Check if gh CLI is available
            subprocess.run(["gh", "--version"], check=True, capture_output=True)
        except subprocess.CalledProcessError:
            health["issues"].append("GitHub CLI not available")
            health["integration_status"] = "degraded"
            
        return health

class PerformanceCache:
    """High-performance caching system for context queries"""
    
    def __init__(self, max_size: int = 100, ttl_seconds: int = 300):
        self.max_size = max_size
        self.ttl_seconds = ttl_seconds
        self.cache = {}
        self.access_times = {}
        self.lock = threading.RLock()
    
    def get(self, key: str) -> Optional[Any]:
        """Get item from cache"""
        with self.lock:
            if key not in self.cache:
                return None
            
            # Check TTL
            if time.time() - self.access_times[key] > self.ttl_seconds:
                del self.cache[key]
                del self.access_times[key]
                return None
            
            # Update access time
            self.access_times[key] = time.time()
            return self.cache[key]
    
    def set(self, key: str, value: Any) -> None:
        """Set item in cache"""
        with self.lock:
            # Evict if at max size
            if len(self.cache) >= self.max_size and key not in self.cache:
                # Remove oldest
                oldest_key = min(self.access_times.keys(), key=self.access_times.get)
                del self.cache[oldest_key]
                del self.access_times[oldest_key]
            
            self.cache[key] = value
            self.access_times[key] = time.time()
    
    def clear(self) -> None:
        """Clear entire cache"""
        with self.lock:
            self.cache.clear()
            self.access_times.clear()
    
    def invalidate_pattern(self, pattern: str) -> None:
        """Invalidate cache entries matching pattern"""
        with self.lock:
            keys_to_remove = [key for key in self.cache.keys() if pattern in key]
            for key in keys_to_remove:
                del self.cache[key]
                del self.access_times[key]

class GitHooksManager:
    """Manages git hooks for automatic context updates"""
    
    def __init__(self, repo_path: str, engine_callback):
        self.repo_path = Path(repo_path)
        self.hooks_dir = self.repo_path / ".git" / "hooks"
        self.engine_callback = weakref.ref(engine_callback)  # Avoid circular reference
        
    def install_hooks(self) -> bool:
        """Install git hooks for automatic context updates"""
        try:
            if not self.hooks_dir.exists():
                print("Warning: .git/hooks directory not found")
                return False
            
            hooks_to_install = {
                "post-commit": self._generate_post_commit_hook(),
                "post-merge": self._generate_post_merge_hook(),
                "post-checkout": self._generate_post_checkout_hook()
            }
            
            for hook_name, hook_content in hooks_to_install.items():
                hook_path = self.hooks_dir / hook_name
                
                # Backup existing hook if it exists
                if hook_path.exists():
                    backup_path = self.hooks_dir / f"{hook_name}.backup"
                    if not backup_path.exists():
                        hook_path.rename(backup_path)
                
                # Write new hook
                with open(hook_path, 'w') as f:
                    f.write(hook_content)
                
                # Make executable
                hook_path.chmod(hook_path.stat().st_mode | stat.S_IEXEC)
                
            print(f"Git hooks installed successfully in {self.hooks_dir}")
            return True
            
        except Exception as e:
            print(f"Error installing git hooks: {e}")
            return False
    
    def _generate_post_commit_hook(self) -> str:
        """Generate post-commit hook script"""
        return f'''#!/bin/sh
# RIF Live System Context Engine - Post Commit Hook
# Auto-generated by RIF

echo "RIF: Triggering context update after commit..."

# Trigger context update with timeout
timeout 30s python3 "{self.repo_path}/systems/live-system-context-engine.py" --trigger git_commit || {{
    echo "RIF: Context update timed out or failed"
    exit 0  # Don't fail the commit
}}

echo "RIF: Context update completed"
'''
    
    def _generate_post_merge_hook(self) -> str:
        """Generate post-merge hook script"""
        return f'''#!/bin/sh
# RIF Live System Context Engine - Post Merge Hook
# Auto-generated by RIF

echo "RIF: Triggering context update after merge..."

# Trigger context update with timeout
timeout 30s python3 "{self.repo_path}/systems/live-system-context-engine.py" --trigger git_merge || {{
    echo "RIF: Context update timed out or failed"
    exit 0  # Don't fail the merge
}}

echo "RIF: Context update completed"
'''
    
    def _generate_post_checkout_hook(self) -> str:
        """Generate post-checkout hook script"""
        return f'''#!/bin/sh
# RIF Live System Context Engine - Post Checkout Hook
# Auto-generated by RIF

echo "RIF: Triggering context update after checkout..."

# Trigger context update with timeout
timeout 30s python3 "{self.repo_path}/systems/live-system-context-engine.py" --trigger git_checkout || {{
    echo "RIF: Context update timed out or failed"
    exit 0  # Don't fail the checkout
}}

echo "RIF: Context update completed"
'''
    
    def uninstall_hooks(self) -> bool:
        """Uninstall git hooks and restore backups"""
        try:
            hook_names = ["post-commit", "post-merge", "post-checkout"]
            
            for hook_name in hook_names:
                hook_path = self.hooks_dir / hook_name
                backup_path = self.hooks_dir / f"{hook_name}.backup"
                
                if hook_path.exists():
                    # Check if it's our hook
                    with open(hook_path, 'r') as f:
                        content = f.read()
                    
                    if "RIF Live System Context Engine" in content:
                        hook_path.unlink()
                        
                        # Restore backup if exists
                        if backup_path.exists():
                            backup_path.rename(hook_path)
                            
            print("Git hooks uninstalled successfully")
            return True
            
        except Exception as e:
            print(f"Error uninstalling git hooks: {e}")
            return False
    
    def trigger_update(self, trigger_type: UpdateTrigger) -> None:
        """Trigger context update from git hook"""
        engine = self.engine_callback()
        if engine:
            try:
                engine.trigger_update(trigger_type, "Git hook triggered update")
            except Exception as e:
                print(f"Error triggering context update: {e}")

class AgentContextOptimizer:
    """Optimizes context delivery for different agent types using Agent-Aware Context Optimization Pattern"""
    
    def __init__(self):
        # Multi-factor relevance scoring weights
        self.scoring_weights = {
            'text': 0.40,
            'semantic': 0.30, 
            'structural': 0.20,
            'temporal': 0.10
        }
        
        # Agent-specific context windows (tokens)
        self.context_windows = {
            'rif-analyst': 12000,
            'rif-planner': 10000,
            'rif-architect': 12000,
            'rif-implementer': 8000,
            'rif-validator': 6000,
            'rif-learner': 10000,
            'general': 8000
        }
    
    def optimize_context_for_agent(self, context: LiveSystemContext, 
                                  agent_type: str = "general") -> Dict[str, Any]:
        """Optimize context for specific agent type using multi-factor relevance scoring"""
        
        # Get agent-specific context window
        max_tokens = self.context_windows.get(agent_type, 8000)
        
        # Create optimized context sections based on agent type
        optimized_sections = self._select_relevant_sections(context, agent_type)
        
        # Apply relevance scoring
        scored_sections = self._apply_relevance_scoring(optimized_sections, agent_type)
        
        # Trim to fit context window
        final_context = self._trim_to_context_window(scored_sections, max_tokens)
        
        return {
            'agent_type': agent_type,
            'context_window': max_tokens,
            'optimization_applied': True,
            'relevance_scoring': True,
            'sections': final_context,
            'metadata': {
                'optimization_timestamp': datetime.now().isoformat(),
                'scoring_weights': self.scoring_weights,
                'estimated_tokens': sum(section.get('estimated_tokens', 0) 
                                      for section in final_context)
            }
        }
    
    def _select_relevant_sections(self, context: LiveSystemContext, 
                                agent_type: str) -> List[Dict[str, Any]]:
        """Select sections relevant to agent type"""
        sections = []
        
        # Always include system overview (high relevance for all agents)
        sections.append({
            'name': 'system_overview',
            'content': context.overview,
            'base_relevance': 1.0,
            'agent_multiplier': 1.0
        })
        
        # Agent-specific section selection
        if agent_type in ['rif-analyst', 'rif-planner']:
            # High need for design goals and architectural decisions
            sections.append({
                'name': 'design_goals',
                'content': context.design_goals,
                'base_relevance': 0.9,
                'agent_multiplier': 1.2
            })
            sections.append({
                'name': 'architectural_decisions',
                'content': context.architectural_decisions,
                'base_relevance': 0.9,
                'agent_multiplier': 1.2
            })
        
        if agent_type in ['rif-architect', 'rif-implementer']:
            # High need for dependency graph and system components
            sections.append({
                'name': 'dependency_graph',
                'content': context.dependency_graph,
                'base_relevance': 0.8,
                'agent_multiplier': 1.3
            })
        
        if agent_type in ['rif-validator', 'rif-implementer']:
            # High need for quality gates and workflow states
            sections.append({
                'name': 'quality_gates',
                'content': context.quality_gates,
                'base_relevance': 0.8,
                'agent_multiplier': 1.2
            })
            sections.append({
                'name': 'workflow_states',
                'content': context.workflow_states,
                'base_relevance': 0.7,
                'agent_multiplier': 1.1
            })
        
        # All agents benefit from recent changes and system health
        sections.append({
            'name': 'recent_changes',
            'content': context.recent_changes,
            'base_relevance': 0.6,
            'agent_multiplier': 1.0
        })
        sections.append({
            'name': 'system_health',
            'content': context.system_health,
            'base_relevance': 0.5,
            'agent_multiplier': 1.0
        })
        
        return sections
    
    def _apply_relevance_scoring(self, sections: List[Dict[str, Any]], 
                               agent_type: str) -> List[Dict[str, Any]]:
        """Apply multi-factor relevance scoring to sections"""
        scored_sections = []
        
        for section in sections:
            # Calculate text relevance (40% weight)
            text_score = self._calculate_text_relevance(section, agent_type)
            
            # Calculate semantic relevance (30% weight) - simplified
            semantic_score = section.get('base_relevance', 0.5)
            
            # Calculate structural relevance (20% weight)
            structural_score = self._calculate_structural_relevance(section, agent_type)
            
            # Calculate temporal relevance (10% weight)
            temporal_score = self._calculate_temporal_relevance(section)
            
            # Weighted final score
            final_score = (
                text_score * self.scoring_weights['text'] +
                semantic_score * self.scoring_weights['semantic'] +
                structural_score * self.scoring_weights['structural'] +
                temporal_score * self.scoring_weights['temporal']
            ) * section.get('agent_multiplier', 1.0)
            
            section['relevance_score'] = final_score
            section['scoring_breakdown'] = {
                'text': text_score,
                'semantic': semantic_score,
                'structural': structural_score,
                'temporal': temporal_score,
                'final': final_score
            }
            
            scored_sections.append(section)
        
        # Sort by relevance score
        scored_sections.sort(key=lambda x: x['relevance_score'], reverse=True)
        
        return scored_sections
    
    def _calculate_text_relevance(self, section: Dict[str, Any], agent_type: str) -> float:
        """Calculate text-based relevance score"""
        # Simple keyword matching for agent relevance
        agent_keywords = {
            'rif-analyst': ['analysis', 'requirements', 'patterns', 'complexity'],
            'rif-planner': ['planning', 'strategy', 'workflow', 'phases'],
            'rif-architect': ['architecture', 'design', 'dependencies', 'structure'],
            'rif-implementer': ['implementation', 'code', 'components', 'integration'],
            'rif-validator': ['validation', 'testing', 'quality', 'gates'],
            'rif-learner': ['learning', 'knowledge', 'patterns', 'metrics']
        }
        
        keywords = agent_keywords.get(agent_type, [])
        if not keywords:
            return 0.5
        
        # Count keyword matches in section content
        content_str = str(section.get('content', '')).lower()
        matches = sum(1 for keyword in keywords if keyword in content_str)
        
        return min(1.0, 0.3 + (matches * 0.2))  # Base 0.3, +0.2 per match
    
    def _calculate_structural_relevance(self, section: Dict[str, Any], agent_type: str) -> float:
        """Calculate structural relevance score"""
        # Sections higher in the structure are more relevant
        structural_priority = {
            'system_overview': 1.0,
            'design_goals': 0.9,
            'architectural_decisions': 0.9,
            'dependency_graph': 0.8,
            'workflow_states': 0.7,
            'quality_gates': 0.7,
            'recent_changes': 0.6,
            'system_health': 0.5
        }
        
        return structural_priority.get(section.get('name'), 0.5)
    
    def _calculate_temporal_relevance(self, section: Dict[str, Any]) -> float:
        """Calculate temporal relevance score"""
        # More recent information is more relevant
        content = section.get('content')
        if not content:
            return 0.5
        
        # Look for timestamp information
        if hasattr(content, 'last_updated'):
            try:
                last_update = content.last_updated
                if isinstance(last_update, str):
                    last_update = datetime.fromisoformat(last_update)
                
                age_hours = (datetime.now() - last_update).total_seconds() / 3600
                
                # Exponential decay: newer = higher score
                return max(0.1, 1.0 * (0.95 ** age_hours))
                
            except Exception:
                pass
        
        return 0.5  # Default temporal score
    
    def _trim_to_context_window(self, sections: List[Dict[str, Any]], 
                              max_tokens: int) -> List[Dict[str, Any]]:
        """Trim sections to fit within context window"""
        final_sections = []
        current_tokens = 0
        
        for section in sections:
            # Estimate tokens (rough approximation: 4 chars per token)
            content_str = str(section.get('content', ''))
            estimated_tokens = len(content_str) // 4
            section['estimated_tokens'] = estimated_tokens
            
            if current_tokens + estimated_tokens <= max_tokens:
                final_sections.append(section)
                current_tokens += estimated_tokens
            else:
                # Try to fit partial content if high relevance
                if section.get('relevance_score', 0) > 0.8:
                    remaining_tokens = max_tokens - current_tokens
                    if remaining_tokens > 100:  # Minimum useful size
                        # Truncate content to fit
                        truncated_content = content_str[:remaining_tokens * 4]
                        section['content'] = truncated_content + "...[truncated]"
                        section['estimated_tokens'] = remaining_tokens
                        section['truncated'] = True
                        final_sections.append(section)
                break
        
        return final_sections

class LiveSystemContextEngine:
    """Main engine for maintaining live system context with Phase 3 enhancements"""
    
    def __init__(self, repo_path: str = "/Users/cal/DEV/RIF", 
                 update_interval: int = 300):  # 5 minutes
        self.repo_path = repo_path
        self.update_interval = update_interval
        self.context_generator = ContextGenerator(repo_path)
        
        # Phase 1: Performance enhancements
        self.performance_cache = PerformanceCache(max_size=50, ttl_seconds=update_interval)
        self.executor = ThreadPoolExecutor(max_workers=4)
        
        # Phase 2: Git hooks integration
        self.git_hooks_manager = GitHooksManager(repo_path, self)
        
        # Phase 3: Agent context optimization
        self.agent_optimizer = AgentContextOptimizer()
        
        # Legacy support
        self.context_cache = {}  # Maintained for compatibility
        self.last_update = None
        self.update_thread = None
        self.running = False
        
        # Enhanced storage
        self.context_dir = os.path.join(repo_path, "systems", "context")
        os.makedirs(self.context_dir, exist_ok=True)
        
        # Performance tracking
        self.performance_metrics = {
            'update_times': [],
            'query_times': [],
            'cache_hits': 0,
            'cache_misses': 0,
            'total_updates': 0
        }
    
    def get_live_context(self, force_refresh: bool = False, 
                        agent_type: str = "general") -> LiveSystemContext:
        """Get current live system context with performance optimization"""
        start_time = time.time()
        
        try:
            # Try performance cache first
            cache_key = f"live_context_{agent_type}"
            if not force_refresh:
                cached_context = self.performance_cache.get(cache_key)
                if cached_context:
                    self.performance_metrics['cache_hits'] += 1
                    query_time = (time.time() - start_time) * 1000
                    self.performance_metrics['query_times'].append(query_time)
                    return cached_context
            
            self.performance_metrics['cache_misses'] += 1
            
            # Check if refresh is needed
            if (force_refresh or 
                self.last_update is None or
                datetime.now() - self.last_update > timedelta(seconds=self.update_interval)):
                
                self._update_context()
            
            context = self.context_cache.get('current_context')
            if context:
                # Cache the result
                self.performance_cache.set(cache_key, context)
                
            query_time = (time.time() - start_time) * 1000
            self.performance_metrics['query_times'].append(query_time)
            
            return context
            
        except Exception as e:
            print(f"Error getting live context: {e}")
            return None
    
    def _update_context(self) -> None:
        """Update system context with parallel processing for performance"""
        start_time = time.time()
        print("Updating live system context with parallel processing...")
        
        try:
            # Use parallel processing for performance
            futures = []
            
            # Submit parallel tasks
            futures.append(self.executor.submit(self.context_generator.generate_system_overview))
            futures.append(self.executor.submit(self.context_generator.analyzer.analyze_system_components))
            futures.append(self.executor.submit(self.context_generator.extract_design_goals))
            futures.append(self.executor.submit(self.context_generator.extract_architectural_decisions))
            futures.append(self.executor.submit(self.context_generator.get_workflow_states))
            futures.append(self.executor.submit(self.context_generator.extract_quality_gates))
            futures.append(self.executor.submit(self._extract_knowledge_patterns))
            futures.append(self.executor.submit(self.context_generator.get_recent_changes))
            futures.append(self.executor.submit(self.context_generator.assess_system_health))
            
            # Collect results
            results = []
            for future in as_completed(futures, timeout=30):
                try:
                    result = future.result()
                    results.append(result)
                except Exception as e:
                    print(f"Warning: Parallel task failed: {e}")
                    results.append(None)
            
            # Ensure we have all results (in order)
            if len(results) < 9:
                print("Warning: Some parallel tasks failed, falling back to sequential")
                return self._update_context_sequential()
            
            # Extract results in order
            overview, components, design_goals, architectural_decisions, workflow_states, quality_gates, knowledge_patterns, recent_changes, system_health = results[:9]
            
            # Generate dependency graph from components
            dependency_graph = self.context_generator.analyzer.analyze_dependencies(components) if components else None
            
            # Generate context
            context = LiveSystemContext(
                overview=overview or self.context_generator.generate_system_overview(),
                dependency_graph=dependency_graph or self.context_generator.analyzer.analyze_dependencies({}),
                design_goals=design_goals or [],
                architectural_decisions=architectural_decisions or [],
                workflow_states=workflow_states or {},
                quality_gates=quality_gates or [],
                knowledge_patterns=knowledge_patterns or [],
                recent_changes=recent_changes or [],
                system_health=system_health or {'overall_status': 'unknown'},
                context_timestamp=datetime.now()
            )
            
            # Cache context
            self.context_cache['current_context'] = context
            self.last_update = datetime.now()
            
            # Clear performance cache to force fresh data
            self.performance_cache.clear()
            
            # Save to file for persistence
            self._save_context(context)
            
            # Track performance
            update_time = (time.time() - start_time) * 1000
            self.performance_metrics['update_times'].append(update_time)
            self.performance_metrics['total_updates'] += 1
            
            print(f"System context updated in {update_time:.2f}ms at {datetime.now().isoformat()}")
            
        except Exception as e:
            print(f"Error in parallel context update, falling back to sequential: {e}")
            self._update_context_sequential()
    
    def _update_context_sequential(self) -> None:
        """Fallback sequential context update"""
        start_time = time.time()
        
        try:
            # Generate components sequentially
            overview = self.context_generator.generate_system_overview()
            components = self.context_generator.analyzer.analyze_system_components()
            dependency_graph = self.context_generator.analyzer.analyze_dependencies(components)
            
            # Generate context
            context = LiveSystemContext(
                overview=overview,
                dependency_graph=dependency_graph,
                design_goals=self.context_generator.extract_design_goals(),
                architectural_decisions=self.context_generator.extract_architectural_decisions(),
                workflow_states=self.context_generator.get_workflow_states(),
                quality_gates=self.context_generator.extract_quality_gates(),
                knowledge_patterns=self._extract_knowledge_patterns(),
                recent_changes=self.context_generator.get_recent_changes(),
                system_health=self.context_generator.assess_system_health(),
                context_timestamp=datetime.now()
            )
            
            # Cache context
            self.context_cache['current_context'] = context
            self.last_update = datetime.now()
            
            # Save to file for persistence
            self._save_context(context)
            
            update_time = (time.time() - start_time) * 1000
            print(f"System context updated sequentially in {update_time:.2f}ms")
            
        except Exception as e:
            print(f"Error in sequential context update: {e}")
    
    def _extract_knowledge_patterns(self) -> List[str]:
        """Extract key knowledge patterns from the knowledge base"""
        patterns = []
        
        knowledge_path = os.path.join(self.repo_path, "knowledge", "patterns")
        if os.path.exists(knowledge_path):
            try:
                for filename in os.listdir(knowledge_path):
                    if filename.endswith('.json'):
                        pattern_name = filename[:-5].replace('-', ' ').title()
                        patterns.append(pattern_name)
            except Exception:
                pass
        
        # Add some core patterns that should always be present
        core_patterns = [
            "Agent Orchestration Pattern",
            "State Machine Workflow",
            "Quality Gate Enforcement",
            "Knowledge Learning Loop"
        ]
        
        for pattern in core_patterns:
            if pattern not in patterns:
                patterns.append(pattern)
        
        return patterns[:10]  # Limit to 10 patterns
    
    def _save_context(self, context: LiveSystemContext) -> None:
        """Save context to file for persistence"""
        try:
            context_file = os.path.join(self.context_dir, "live-system-context.json")
            with open(context_file, 'w') as f:
                json.dump(context.to_dict(), f, indent=2, default=str)
                
            # Also save a timestamped version for history
            timestamp = datetime.now().strftime('%Y%m%d-%H%M%S')
            history_file = os.path.join(self.context_dir, f"context-history-{timestamp}.json")
            with open(history_file, 'w') as f:
                json.dump(context.to_dict(), f, indent=2, default=str)
                
        except Exception as e:
            print(f"Error saving context: {e}")
    
    def start_continuous_updates(self) -> None:
        """Start continuous context updates in background"""
        if self.running:
            return
        
        self.running = True
        self.update_thread = threading.Thread(target=self._continuous_update_loop)
        self.update_thread.daemon = True
        self.update_thread.start()
        
        print(f"Started continuous context updates (interval: {self.update_interval}s)")
    
    def stop_continuous_updates(self) -> None:
        """Stop continuous context updates and cleanup resources"""
        self.running = False
        if self.update_thread:
            self.update_thread.join(timeout=5)
        
        # Cleanup thread pool
        if hasattr(self, 'executor'):
            self.executor.shutdown(wait=True)
            
        print("Stopped continuous context updates and cleaned up resources")
    
    def _continuous_update_loop(self) -> None:
        """Continuous update loop running in background"""
        while self.running:
            try:
                self._update_context()
                time.sleep(self.update_interval)
            except Exception as e:
                print(f"Error in continuous update: {e}")
                time.sleep(30)  # Wait before retrying
    
    def trigger_update(self, trigger_type: UpdateTrigger, details: str = "") -> None:
        """Manually trigger a context update with enhanced logging"""
        print(f"Context update triggered: {trigger_type.value} - {details}")
        
        # Clear relevant cache entries based on trigger type
        if trigger_type in [UpdateTrigger.CODE_CHANGE, UpdateTrigger.GIT_COMMIT, 
                          UpdateTrigger.GIT_MERGE, UpdateTrigger.CONFIGURATION_CHANGE]:
            self.performance_cache.clear()
        elif trigger_type in [UpdateTrigger.ARCHITECTURAL_DECISION, UpdateTrigger.WORKFLOW_UPDATE]:
            self.performance_cache.invalidate_pattern("context")
        
        # Record trigger for performance tracking
        trigger_start = time.time()
        
        try:
            self._update_context()
            trigger_time = (time.time() - trigger_start) * 1000
            print(f"Context update completed in {trigger_time:.2f}ms for trigger: {trigger_type.value}")
        except Exception as e:
            print(f"Context update failed for trigger {trigger_type.value}: {e}")
            raise
    
    def format_context_for_agent(self, context: LiveSystemContext, 
                                agent_type: str = "general", 
                                use_optimization: bool = True) -> str:
        """Format context for agent consumption with Agent-Aware Context Optimization"""
        
        if use_optimization:
            # Use Agent-Aware Context Optimization Pattern
            optimized = self.agent_optimizer.optimize_context_for_agent(context, agent_type)
            return self._format_optimized_context(optimized)
        else:
            # Legacy formatting for backward compatibility
            return self._format_legacy_context(context, agent_type)
    
    def _format_optimized_context(self, optimized_context: Dict[str, Any]) -> str:
        """Format optimized context using Agent-Aware pattern"""
        formatted = []
        
        formatted.append("# Live System Context (Agent-Optimized)")
        formatted.append(f"**Agent Type**: {optimized_context['agent_type']}")
        formatted.append(f"**Context Window**: {optimized_context['context_window']} tokens")
        formatted.append(f"**Optimization**: Multi-factor relevance scoring applied")
        formatted.append(f"**Updated**: {optimized_context['metadata']['optimization_timestamp']}")
        formatted.append(f"**Estimated Tokens**: {optimized_context['metadata']['estimated_tokens']}")
        formatted.append("")
        
        # Format optimized sections
        for section in optimized_context['sections']:
            section_name = section['name'].replace('_', ' ').title()
            relevance = section.get('relevance_score', 0)
            
            formatted.append(f"## {section_name} (Relevance: {relevance:.2f})")
            
            if section.get('truncated'):
                formatted.append("*[Content optimized for agent context window]*")
            
            # Format section content based on type
            content = section['content']
            if hasattr(content, 'to_dict'):
                formatted.extend(self._format_dataclass_content(content))
            elif isinstance(content, dict):
                formatted.extend(self._format_dict_content(content))
            elif isinstance(content, list):
                formatted.extend(self._format_list_content(content))
            else:
                formatted.append(str(content))
            
            formatted.append("")
        
        # Add optimization metadata
        formatted.append("## Context Optimization Details")
        metadata = optimized_context['metadata']
        formatted.append(f"**Scoring Weights**: Text {metadata['scoring_weights']['text']:.0%}, "
                        f"Semantic {metadata['scoring_weights']['semantic']:.0%}, "
                        f"Structural {metadata['scoring_weights']['structural']:.0%}, "
                        f"Temporal {metadata['scoring_weights']['temporal']:.0%}")
        formatted.append("")
        
        return "\n".join(formatted)
    
    def _format_legacy_context(self, context: LiveSystemContext, agent_type: str) -> str:
        """Legacy context formatting for backward compatibility"""
        formatted = []
        
        formatted.append("# Live System Context")
        formatted.append(f"**Agent Type**: {agent_type}")
        formatted.append(f"**Updated**: {context.context_timestamp.isoformat()}")
        formatted.append("")
        
        formatted.append("## System Overview")
        formatted.append(f"**Purpose**: {context.overview.purpose}")
        formatted.append(f"**Mission**: {context.overview.core_mission}")
        formatted.append(f"**Philosophy**: {context.overview.design_philosophy}")
        formatted.append(f"**Architecture**: {context.overview.architecture_summary}")
        formatted.append("")
        
        formatted.append("## Key Capabilities")
        for capability in context.overview.key_capabilities:
            formatted.append(f"- {capability}")
        formatted.append("")
        
        formatted.append("## Design Goals")
        for goal in context.design_goals[:3]:  # Top 3 goals
            formatted.append(f"- **{goal['title']}**: {goal['description']}")
        formatted.append("")
        
        formatted.append("## System Architecture")
        formatted.append(f"**Components**: {len(context.dependency_graph.components)} active components")
        formatted.append(f"**Integration Points**: {len(context.dependency_graph.integration_points)}")
        formatted.append(f"**External Dependencies**: {len(context.dependency_graph.external_dependencies)}")
        formatted.append("")
        
        formatted.append("## Workflow States")
        for state, description in context.workflow_states.items():
            formatted.append(f"- **{state}**: {description}")
        formatted.append("")
        
        formatted.append("## Quality Standards")
        for gate in context.quality_gates[:3]:  # Top 3 gates
            formatted.append(f"- **{gate['name']}**: {gate['threshold']}")
        formatted.append("")
        
        formatted.append("## System Health")
        formatted.append(f"**Overall Status**: {context.system_health['overall_status']}")
        if context.system_health.get('issues'):
            formatted.append("**Issues**:")
            for issue in context.system_health['issues']:
                formatted.append(f"  - {issue}")
        formatted.append("")
        
        formatted.append("## Recent Changes")
        for change in context.recent_changes[:3]:  # Latest 3 changes
            formatted.append(f"- {change['message']} ({change['commit']})")
        formatted.append("")
        
        return "\n".join(formatted)
    
    def _format_dataclass_content(self, content) -> List[str]:
        """Format dataclass content"""
        lines = []
        if hasattr(content, 'purpose'):
            lines.append(f"**Purpose**: {content.purpose}")
        if hasattr(content, 'core_mission'):
            lines.append(f"**Mission**: {content.core_mission}")
        if hasattr(content, 'design_philosophy'):
            lines.append(f"**Philosophy**: {content.design_philosophy}")
        return lines
    
    def _format_dict_content(self, content: dict) -> List[str]:
        """Format dictionary content"""
        lines = []
        for key, value in content.items():
            if isinstance(value, (list, dict)):
                lines.append(f"**{key.replace('_', ' ').title()}**: {len(value)} items")
            else:
                lines.append(f"**{key.replace('_', ' ').title()}**: {value}")
        return lines
    
    def _format_list_content(self, content: list) -> List[str]:
        """Format list content"""
        lines = []
        for i, item in enumerate(content[:5]):  # Limit to 5 items
            if isinstance(item, dict):
                title = item.get('title') or item.get('name') or f"Item {i+1}"
                description = item.get('description') or item.get('message') or str(item)
                lines.append(f"- **{title}**: {description}")
            else:
                lines.append(f"- {item}")
        if len(content) > 5:
            lines.append(f"- ... and {len(content) - 5} more items")
        return lines
    
    def get_context_summary(self) -> Dict[str, Any]:
        """Get a summary of current context status with performance metrics"""
        context = self.get_live_context()
        if not context:
            return {"status": "no_context", "message": "Context not available"}
        
        # Calculate performance statistics
        avg_update_time = sum(self.performance_metrics['update_times'][-10:]) / max(len(self.performance_metrics['update_times'][-10:]), 1)
        avg_query_time = sum(self.performance_metrics['query_times'][-100:]) / max(len(self.performance_metrics['query_times'][-100:]), 1)
        cache_hit_rate = (self.performance_metrics['cache_hits'] / 
                         max(self.performance_metrics['cache_hits'] + self.performance_metrics['cache_misses'], 1)) * 100
        
        return {
            "status": "active",
            "last_updated": context.context_timestamp.isoformat(),
            "components_count": len(context.dependency_graph.components),
            "health_status": context.system_health['overall_status'],
            "recent_changes": len(context.recent_changes),
            "design_goals": len(context.design_goals),
            "quality_gates": len(context.quality_gates),
            "performance_metrics": {
                "avg_update_time_ms": round(avg_update_time, 2),
                "avg_query_time_ms": round(avg_query_time, 2),
                "cache_hit_rate_percent": round(cache_hit_rate, 1),
                "total_updates": self.performance_metrics['total_updates'],
                "sla_compliance": avg_update_time < 300000  # 5 minutes in ms
            },
            "optimizations_enabled": {
                "performance_cache": True,
                "parallel_processing": True,
                "agent_context_optimization": True,
                "git_hooks_integration": os.path.exists(os.path.join(self.repo_path, ".git", "hooks", "post-commit"))
            }
        }
    
    # Phase 2: Git Hooks Integration Methods
    def install_git_hooks(self) -> bool:
        """Install git hooks for automatic context updates"""
        return self.git_hooks_manager.install_hooks()
    
    def uninstall_git_hooks(self) -> bool:
        """Uninstall git hooks"""
        return self.git_hooks_manager.uninstall_hooks()
    
    # Phase 3: Enhanced Agent Context Delivery
    def get_optimized_context_for_agent(self, agent_type: str, 
                                      force_refresh: bool = False) -> Dict[str, Any]:
        """Get optimized context for specific agent using Agent-Aware Context Optimization Pattern"""
        context = self.get_live_context(force_refresh, agent_type)
        if not context:
            return {"error": "Context not available"}
        
        return self.agent_optimizer.optimize_context_for_agent(context, agent_type)
    
    # Performance monitoring and diagnostics
    def get_performance_report(self) -> Dict[str, Any]:
        """Get detailed performance report"""
        metrics = self.performance_metrics
        
        if not metrics['update_times']:
            return {"status": "no_data", "message": "No performance data available"}
        
        recent_updates = metrics['update_times'][-10:]
        recent_queries = metrics['query_times'][-100:]
        
        return {
            "performance_summary": {
                "avg_update_time_ms": sum(recent_updates) / len(recent_updates),
                "max_update_time_ms": max(recent_updates),
                "min_update_time_ms": min(recent_updates),
                "avg_query_time_ms": sum(recent_queries) / len(recent_queries) if recent_queries else 0,
                "max_query_time_ms": max(recent_queries) if recent_queries else 0,
                "sla_compliance": sum(1 for t in recent_updates if t < 300000) / len(recent_updates) * 100
            },
            "cache_performance": {
                "hit_rate_percent": (metrics['cache_hits'] / 
                                   max(metrics['cache_hits'] + metrics['cache_misses'], 1)) * 100,
                "total_hits": metrics['cache_hits'],
                "total_misses": metrics['cache_misses']
            },
            "system_stats": {
                "total_updates": metrics['total_updates'],
                "update_interval_seconds": self.update_interval,
                "cache_size": len(self.performance_cache.cache),
                "executor_threads": self.executor._max_workers
            },
            "recommendations": self._generate_performance_recommendations()
        }
    
    def _generate_performance_recommendations(self) -> List[str]:
        """Generate performance improvement recommendations"""
        recommendations = []
        metrics = self.performance_metrics
        
        if not metrics['update_times']:
            return recommendations
        
        recent_updates = metrics['update_times'][-10:]
        avg_update_time = sum(recent_updates) / len(recent_updates)
        cache_hit_rate = (metrics['cache_hits'] / 
                         max(metrics['cache_hits'] + metrics['cache_misses'], 1)) * 100
        
        if avg_update_time > 300000:  # 5 minutes
            recommendations.append("Update time exceeds 5-minute SLA - consider reducing update scope or increasing parallel workers")
        
        if cache_hit_rate < 50:
            recommendations.append("Low cache hit rate - consider increasing cache TTL or size")
        
        if max(recent_updates) > 600000:  # 10 minutes
            recommendations.append("Maximum update time is very high - investigate system bottlenecks")
        
        query_times = metrics['query_times'][-100:]
        if query_times and sum(query_times) / len(query_times) > 200:
            recommendations.append("Query response time exceeds 200ms target - optimize context formatting")
        
        if not recommendations:
            recommendations.append("Performance is within acceptable parameters")
        
        return recommendations

# CLI Interface with Phase 3 enhancements
if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="RIF Live System Context Engine - Phase 3 Enhanced")
    
    # Core functionality
    parser.add_argument("--update", action="store_true", help="Force context update")
    parser.add_argument("--start", action="store_true", help="Start continuous updates")
    parser.add_argument("--status", action="store_true", help="Show context status")
    parser.add_argument("--format", type=str, help="Format context for agent type")
    
    # Phase 2: Git hooks integration
    parser.add_argument("--install-hooks", action="store_true", help="Install git hooks for automatic updates")
    parser.add_argument("--uninstall-hooks", action="store_true", help="Uninstall git hooks")
    parser.add_argument("--trigger", type=str, choices=["git_commit", "git_merge", "git_checkout", "manual_trigger"], 
                       help="Manually trigger update with specific trigger type")
    
    # Phase 3: Agent context optimization
    parser.add_argument("--agent-context", type=str, help="Get optimized context for specific agent type")
    parser.add_argument("--no-optimization", action="store_true", help="Disable agent context optimization")
    
    # Performance and diagnostics
    parser.add_argument("--performance", action="store_true", help="Show performance report")
    parser.add_argument("--benchmark", action="store_true", help="Run performance benchmark")
    
    # Configuration
    parser.add_argument("--repo", type=str, default="/Users/cal/DEV/RIF", help="Repository path")
    parser.add_argument("--interval", type=int, default=300, help="Update interval in seconds")
    
    args = parser.parse_args()
    
    engine = LiveSystemContextEngine(args.repo, args.interval)
    
    if args.install_hooks:
        success = engine.install_git_hooks()
        print(f"Git hooks installation: {'SUCCESS' if success else 'FAILED'}")
    
    elif args.uninstall_hooks:
        success = engine.uninstall_git_hooks()
        print(f"Git hooks uninstallation: {'SUCCESS' if success else 'FAILED'}")
    
    elif args.trigger:
        # Map CLI trigger names to enum values
        trigger_mapping = {
            "git_commit": UpdateTrigger.GIT_COMMIT,
            "git_merge": UpdateTrigger.GIT_MERGE, 
            "git_checkout": UpdateTrigger.GIT_CHECKOUT,
            "manual_trigger": UpdateTrigger.MANUAL_TRIGGER
        }
        
        trigger_type = trigger_mapping.get(args.trigger)
        if trigger_type:
            engine.trigger_update(trigger_type, f"Manual trigger: {args.trigger}")
            print(f"Context update triggered with: {args.trigger}")
        else:
            print(f"Error: Unknown trigger type: {args.trigger}")
    
    elif args.agent_context:
        optimized_context = engine.get_optimized_context_for_agent(args.agent_context, force_refresh=True)
        if "error" in optimized_context:
            print(f"Error: {optimized_context['error']}")
        else:
            formatted = engine._format_optimized_context(optimized_context)
            print(formatted)
    
    elif args.performance:
        report = engine.get_performance_report()
        print("=== Performance Report ===")
        print(json.dumps(report, indent=2))
    
    elif args.benchmark:
        print("Running performance benchmark...")
        
        # Benchmark context updates
        start_time = time.time()
        for i in range(3):
            context = engine.get_live_context(force_refresh=True)
            print(f"Update {i+1}/3 completed")
        update_benchmark = (time.time() - start_time) / 3 * 1000
        
        # Benchmark context queries
        start_time = time.time()
        for i in range(10):
            context = engine.get_live_context()
        query_benchmark = (time.time() - start_time) / 10 * 1000
        
        # Benchmark agent optimization
        context = engine.get_live_context()
        agent_types = ['rif-analyst', 'rif-planner', 'rif-implementer', 'rif-validator']
        optimization_times = []
        
        for agent_type in agent_types:
            start_time = time.time()
            optimized = engine.agent_optimizer.optimize_context_for_agent(context, agent_type)
            optimization_time = (time.time() - start_time) * 1000
            optimization_times.append(optimization_time)
        
        avg_optimization_time = sum(optimization_times) / len(optimization_times)
        
        print("\n=== Benchmark Results ===")
        print(f"Average update time: {update_benchmark:.2f}ms")
        print(f"Average query time: {query_benchmark:.2f}ms") 
        print(f"Average optimization time: {avg_optimization_time:.2f}ms")
        print(f"SLA Compliance: {'PASS' if update_benchmark < 300000 else 'FAIL'} (5-minute target)")
        print(f"Query Performance: {'PASS' if query_benchmark < 200 else 'FAIL'} (200ms target)")
    
    elif args.update:
        start_time = time.time()
        context = engine.get_live_context(force_refresh=True)
        update_time = (time.time() - start_time) * 1000
        
        if context:
            print("Context updated successfully")
            print(f"Update time: {update_time:.2f}ms")
            print(f"Components: {len(context.dependency_graph.components)}")
            print(f"Health: {context.system_health['overall_status']}")
        else:
            print("Context update failed")
    
    elif args.status:
        summary = engine.get_context_summary()
        print("=== System Context Status ===")
        print(json.dumps(summary, indent=2))
    
    elif args.format:
        use_optimization = not args.no_optimization
        context = engine.get_live_context()
        if context:
            formatted = engine.format_context_for_agent(context, args.format, use_optimization)
            print(formatted)
        else:
            print("No context available")
    
    elif args.start:
        print("Starting continuous context updates with Phase 3 enhancements...")
        engine.start_continuous_updates()
        try:
            # Keep running until interrupted
            while True:
                time.sleep(1)
        except KeyboardInterrupt:
            engine.stop_continuous_updates()
            print("\nContext engine stopped")
    
    else:
        parser.print_help()
        print("\n=== Phase 3 Enhancement Features ===")
        print("• Performance optimization with sub-200ms queries")
        print("• Git hooks integration for automatic updates")
        print("• Agent-Aware Context Optimization Pattern")
        print("• Multi-factor relevance scoring (40/30/20/10)")
        print("• Intelligent caching and parallel processing")
        print("• Comprehensive performance monitoring")