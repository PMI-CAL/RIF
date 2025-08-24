#!/usr/bin/env python3
"""
Agent-Specific Context Templates and Formatting Optimization
Issue #123: DPIBS Development Phase 1

Implements optimized context templates for each RIF agent type with:
- Agent-specific formatting and content organization
- Context window optimization per agent role
- Dynamic template adaptation based on task complexity
- Performance-optimized rendering (<50ms formatting time)
- Intelligent content prioritization and truncation

Based on RIF-Architect specifications for Context Intelligence Platform.
"""

import json
import time
import re
from datetime import datetime
from typing import Dict, List, Any, Optional, Tuple
from dataclasses import dataclass, asdict
from enum import Enum
from abc import ABC, abstractmethod

# Import context optimization components
from import_utils import import_context_optimization_engine
context_imports = import_context_optimization_engine()
AgentType = context_imports['AgentType']
ContextType = context_imports['ContextType']
ContextItem = context_imports['ContextItem']
SystemContext = context_imports['SystemContext']
AgentContext = context_imports['AgentContext']

class TemplateSection(Enum):
    """Template sections for content organization"""
    SYSTEM_OVERVIEW = "system_overview"
    CLAUDE_CAPABILITIES = "claude_capabilities"
    TASK_CONTEXT = "task_context"
    RELEVANT_KNOWLEDGE = "relevant_knowledge"
    IMPLEMENTATION_PATTERNS = "implementation_patterns"
    QUALITY_GUIDELINES = "quality_guidelines"
    PERFORMANCE_TARGETS = "performance_targets"
    SUCCESS_CRITERIA = "success_criteria"
    CONTEXT_METADATA = "context_metadata"

class ContentPriority(Enum):
    """Content priority levels for truncation decisions"""
    CRITICAL = 1    # Never truncate
    HIGH = 2        # Truncate only if necessary
    MEDIUM = 3      # Can be summarized
    LOW = 4         # Can be omitted

@dataclass
class TemplateSection:
    """Individual template section with formatting rules"""
    section_type: TemplateSection
    title: str
    content_formatter: str  # Format string or template
    priority: ContentPriority
    max_chars: Optional[int] = None
    collapse_threshold: int = 200  # Chars before collapsing
    agent_relevance: Dict[AgentType, float] = None

@dataclass
class FormattingMetrics:
    """Metrics for template formatting performance"""
    agent_type: AgentType
    template_name: str
    formatting_time_ms: float
    input_size_chars: int
    output_size_chars: int
    sections_included: int
    sections_truncated: int
    truncation_applied: bool

class BaseAgentTemplate(ABC):
    """Base class for agent-specific context templates"""
    
    def __init__(self, agent_type: AgentType):
        self.agent_type = agent_type
        self.max_context_chars = self._get_context_limit()
        self.formatting_metrics = []
        
    def _get_context_limit(self) -> int:
        """Get context character limit for this agent type"""
        limits = {
            AgentType.ANALYST: 4000,
            AgentType.PLANNER: 5000,
            AgentType.ARCHITECT: 6000,
            AgentType.IMPLEMENTER: 3500,
            AgentType.VALIDATOR: 4500,
            AgentType.LEARNER: 3000,
            AgentType.PR_MANAGER: 2500,
            AgentType.ERROR_ANALYST: 4000,
            AgentType.SHADOW_AUDITOR: 5000,
            AgentType.PROJECTGEN: 4500,
        }
        return limits.get(self.agent_type, 3500)
    
    @abstractmethod
    def get_template_sections(self) -> List[TemplateSection]:
        """Get ordered template sections for this agent type"""
        pass
    
    @abstractmethod
    def format_section_content(self, section: TemplateSection, 
                              content: Any, context: AgentContext) -> str:
        """Format content for specific section"""
        pass
    
    def render_context(self, agent_context: AgentContext) -> str:
        """Render complete context using agent-specific template"""
        start_time = time.time()
        
        sections = self.get_template_sections()
        rendered_sections = []
        total_chars = 0
        sections_included = 0
        sections_truncated = 0
        
        # First pass: render all sections
        section_content = {}
        for section in sections:
            content = self.format_section_content(section, None, agent_context)
            if content:
                section_content[section.section_type] = content
        
        # Second pass: apply prioritization and truncation
        remaining_chars = self.max_context_chars
        
        # Critical sections first
        for section in sections:
            if section.priority == ContentPriority.CRITICAL:
                content = section_content.get(section.section_type, "")
                if content:
                    rendered_sections.append(content)
                    total_chars += len(content)
                    remaining_chars -= len(content)
                    sections_included += 1
        
        # High priority sections
        for section in sections:
            if section.priority == ContentPriority.HIGH and remaining_chars > 500:
                content = section_content.get(section.section_type, "")
                if content:
                    if len(content) <= remaining_chars:
                        rendered_sections.append(content)
                        total_chars += len(content)
                        remaining_chars -= len(content)
                        sections_included += 1
                    else:
                        # Truncate high priority content
                        truncated = self._truncate_content(content, remaining_chars)
                        rendered_sections.append(truncated)
                        total_chars += len(truncated)
                        remaining_chars -= len(truncated)
                        sections_included += 1
                        sections_truncated += 1
        
        # Medium and low priority sections
        for priority in [ContentPriority.MEDIUM, ContentPriority.LOW]:
            for section in sections:
                if section.priority == priority and remaining_chars > 200:
                    content = section_content.get(section.section_type, "")
                    if content:
                        if len(content) <= remaining_chars:
                            rendered_sections.append(content)
                            total_chars += len(content)
                            remaining_chars -= len(content)
                            sections_included += 1
                        elif priority == ContentPriority.MEDIUM and remaining_chars > 100:
                            # Summarize medium priority content
                            summarized = self._summarize_content(content, remaining_chars)
                            rendered_sections.append(summarized)
                            total_chars += len(summarized)
                            remaining_chars -= len(summarized)
                            sections_included += 1
                            sections_truncated += 1
        
        # Join sections
        final_context = "\n\n".join(rendered_sections)
        
        # Record metrics
        formatting_time = (time.time() - start_time) * 1000
        metrics = FormattingMetrics(
            agent_type=self.agent_type,
            template_name=self.__class__.__name__,
            formatting_time_ms=formatting_time,
            input_size_chars=sum(len(agent_context.relevant_knowledge[i].content) 
                                for i in range(len(agent_context.relevant_knowledge))),
            output_size_chars=len(final_context),
            sections_included=sections_included,
            sections_truncated=sections_truncated,
            truncation_applied=sections_truncated > 0
        )
        self.formatting_metrics.append(metrics)
        
        return final_context
    
    def _truncate_content(self, content: str, max_chars: int) -> str:
        """Intelligently truncate content preserving important information"""
        if len(content) <= max_chars:
            return content
        
        # Try to truncate at sentence boundaries
        sentences = content.split('. ')
        truncated = ""
        
        for sentence in sentences:
            if len(truncated + sentence + ". ") <= max_chars - 20:  # Reserve space for ellipsis
                truncated += sentence + ". "
            else:
                break
        
        if truncated:
            return truncated.rstrip() + "..."
        else:
            # Fallback: hard truncate with ellipsis
            return content[:max_chars-3] + "..."
    
    def _summarize_content(self, content: str, max_chars: int) -> str:
        """Create summary of content to fit within character limit"""
        if len(content) <= max_chars:
            return content
        
        # Extract key phrases and bullet points
        lines = content.split('\n')
        important_lines = []
        
        for line in lines:
            line = line.strip()
            if (line.startswith('•') or line.startswith('-') or 
                line.startswith('*') or ':' in line or 
                any(keyword in line.lower() for keyword in ['key', 'important', 'critical', 'must'])):
                important_lines.append(line)
        
        if important_lines:
            summary = '\n'.join(important_lines)
            if len(summary) <= max_chars:
                return summary
        
        # Fallback to truncation
        return self._truncate_content(content, max_chars)

class AnalystTemplate(BaseAgentTemplate):
    """Context template optimized for RIF-Analyst"""
    
    def __init__(self):
        super().__init__(AgentType.ANALYST)
    
    def get_template_sections(self) -> List[TemplateSection]:
        return [
            TemplateSection(
                TemplateSection.CLAUDE_CAPABILITIES,
                "Claude Code Capabilities",
                "claude_capabilities_format",
                ContentPriority.CRITICAL,
                max_chars=400
            ),
            TemplateSection(
                TemplateSection.TASK_CONTEXT,
                "Analysis Task Context",
                "task_context_format", 
                ContentPriority.CRITICAL,
                max_chars=600
            ),
            TemplateSection(
                TemplateSection.SYSTEM_OVERVIEW,
                "System Overview",
                "system_overview_format",
                ContentPriority.HIGH,
                max_chars=500
            ),
            TemplateSection(
                TemplateSection.RELEVANT_KNOWLEDGE,
                "Relevant Knowledge & Similar Issues",
                "knowledge_format",
                ContentPriority.HIGH,
                max_chars=800
            ),
            TemplateSection(
                TemplateSection.SUCCESS_CRITERIA,
                "Analysis Success Criteria", 
                "success_criteria_format",
                ContentPriority.MEDIUM,
                max_chars=300
            ),
            TemplateSection(
                TemplateSection.CONTEXT_METADATA,
                "Context Optimization Info",
                "metadata_format",
                ContentPriority.LOW,
                max_chars=200
            )
        ]
    
    def format_section_content(self, section: TemplateSection, 
                              content: Any, context: AgentContext) -> str:
        if section.section_type == TemplateSection.CLAUDE_CAPABILITIES:
            claude_items = [item for item in context.relevant_knowledge 
                           if item.type == ContextType.CLAUDE_CODE_CAPABILITIES]
            if claude_items:
                return f"## 🔧 Claude Code Capabilities\n\n{claude_items[0].content}\n\n" \
                      f"**Key Point**: Focus on analysis using Claude's built-in tools, not orchestration."
        
        elif section.section_type == TemplateSection.TASK_CONTEXT:
            task_desc = context.task_specific_context.get('description', 'No description provided')
            complexity = context.task_specific_context.get('complexity', 'unknown')
            return f"## 🎯 Analysis Task\n\n" \
                   f"**Description**: {task_desc}\n\n" \
                   f"**Complexity**: {complexity.upper()}\n\n" \
                   f"**Analysis Objective**: Extract requirements, assess complexity, identify patterns."
        
        elif section.section_type == TemplateSection.SYSTEM_OVERVIEW:
            return f"## 🏗️ System Context\n\n" \
                   f"**System**: {context.system_context.overview}\n\n" \
                   f"**Purpose**: {context.system_context.purpose}\n\n" \
                   f"**Design Goals**: {', '.join(context.system_context.design_goals[:3])}"
        
        elif section.section_type == TemplateSection.RELEVANT_KNOWLEDGE:
            knowledge_content = []
            similar_items = [item for item in context.relevant_knowledge 
                           if item.type == ContextType.SIMILAR_ISSUES]
            pattern_items = [item for item in context.relevant_knowledge 
                           if item.type == ContextType.IMPLEMENTATION_PATTERNS]
            
            if similar_items:
                knowledge_content.append(f"**Similar Issues**: {similar_items[0].content}")
            if pattern_items:
                knowledge_content.append(f"**Implementation Patterns**: {pattern_items[0].content}")
            
            if knowledge_content:
                return f"## 📚 Relevant Knowledge\n\n" + "\n\n".join(knowledge_content)
        
        elif section.section_type == TemplateSection.SUCCESS_CRITERIA:
            return f"## ✅ Analysis Success Criteria\n\n" \
                   f"• Requirements clearly extracted and documented\n" \
                   f"• Complexity assessment completed with justification\n" \
                   f"• Similar patterns identified from knowledge base\n" \
                   f"• Dependencies mapped and validated\n" \
                   f"• Next steps clearly defined for planning phase"
        
        elif section.section_type == TemplateSection.CONTEXT_METADATA:
            return f"## 📊 Context Optimization\n\n" \
                   f"**Window Utilization**: {context.context_window_utilization:.1%}\n" \
                   f"**Knowledge Items**: {len(context.relevant_knowledge)}\n" \
                   f"**Agent Type**: {context.agent_type.value}"
        
        return ""

class ImplementerTemplate(BaseAgentTemplate):
    """Context template optimized for RIF-Implementer"""
    
    def __init__(self):
        super().__init__(AgentType.IMPLEMENTER)
    
    def get_template_sections(self) -> List[TemplateSection]:
        return [
            TemplateSection(
                TemplateSection.CLAUDE_CAPABILITIES,
                "Claude Code Tools",
                "claude_tools_format",
                ContentPriority.CRITICAL,
                max_chars=350
            ),
            TemplateSection(
                TemplateSection.IMPLEMENTATION_PATTERNS,
                "Implementation Patterns",
                "patterns_format",
                ContentPriority.CRITICAL,
                max_chars=700
            ),
            TemplateSection(
                TemplateSection.TASK_CONTEXT,
                "Implementation Task",
                "impl_task_format",
                ContentPriority.CRITICAL,
                max_chars=500
            ),
            TemplateSection(
                TemplateSection.PERFORMANCE_TARGETS,
                "Performance Requirements",
                "performance_format", 
                ContentPriority.HIGH,
                max_chars=400
            ),
            TemplateSection(
                TemplateSection.QUALITY_GUIDELINES,
                "Quality Guidelines",
                "quality_format",
                ContentPriority.HIGH,
                max_chars=400
            ),
            TemplateSection(
                TemplateSection.RELEVANT_KNOWLEDGE,
                "Technical Knowledge",
                "tech_knowledge_format",
                ContentPriority.MEDIUM,
                max_chars=600
            ),
            TemplateSection(
                TemplateSection.CONTEXT_METADATA,
                "Implementation Metadata",
                "impl_metadata_format",
                ContentPriority.LOW,
                max_chars=200
            )
        ]
    
    def format_section_content(self, section: TemplateSection,
                              content: Any, context: AgentContext) -> str:
        if section.section_type == TemplateSection.CLAUDE_CAPABILITIES:
            claude_items = [item for item in context.relevant_knowledge 
                           if item.type == ContextType.CLAUDE_CODE_CAPABILITIES]
            if claude_items:
                return f"## 🛠️ Claude Code Tools\n\n{claude_items[0].content}\n\n" \
                      f"**Implementation Focus**: Use Read/Write/Edit/MultiEdit, Bash, Git integration efficiently."
        
        elif section.section_type == TemplateSection.IMPLEMENTATION_PATTERNS:
            pattern_items = [item for item in context.relevant_knowledge 
                           if item.type == ContextType.IMPLEMENTATION_PATTERNS]
            if pattern_items:
                return f"## 🏗️ Implementation Patterns\n\n{pattern_items[0].content}"
        
        elif section.section_type == TemplateSection.TASK_CONTEXT:
            task_desc = context.task_specific_context.get('description', 'No description provided')
            complexity = context.task_specific_context.get('complexity', 'unknown')
            return f"## 🎯 Implementation Task\n\n" \
                   f"**Description**: {task_desc}\n\n" \
                   f"**Complexity**: {complexity.upper()}\n\n" \
                   f"**Implementation Focus**: Write production-ready code with comprehensive testing."
        
        elif section.section_type == TemplateSection.PERFORMANCE_TARGETS:
            return f"## ⚡ Performance Requirements\n\n" \
                   f"• **Response Time**: Sub-200ms average, <500ms P99\n" \
                   f"• **Concurrency**: Support 10+ simultaneous requests\n" \
                   f"• **Reliability**: 99.9% availability with graceful degradation\n" \
                   f"• **Cache Performance**: 85%+ hit rates for optimal performance"
        
        elif section.section_type == TemplateSection.QUALITY_GUIDELINES:
            return f"## 🔍 Quality Guidelines\n\n" \
                   f"• **Testing**: >80% coverage, comprehensive unit and integration tests\n" \
                   f"• **Code Quality**: Clean code, proper error handling, type hints\n" \
                   f"• **Performance**: Profile critical paths, optimize for requirements\n" \
                   f"• **Documentation**: Clear docstrings and implementation comments"
        
        elif section.section_type == TemplateSection.RELEVANT_KNOWLEDGE:
            arch_items = [item for item in context.relevant_knowledge 
                         if item.type == ContextType.ARCHITECTURAL_DECISIONS]
            dep_items = [item for item in context.relevant_knowledge 
                        if item.type == ContextType.DEPENDENCY_INFO]
            
            knowledge_parts = []
            if arch_items:
                knowledge_parts.append(f"**Architecture**: {arch_items[0].content}")
            if dep_items:
                knowledge_parts.append(f"**Dependencies**: {dep_items[0].content}")
            
            if knowledge_parts:
                return f"## 📚 Technical Knowledge\n\n" + "\n\n".join(knowledge_parts)
        
        elif section.section_type == TemplateSection.CONTEXT_METADATA:
            return f"## 📊 Implementation Context\n\n" \
                   f"**Context Window**: {context.context_window_utilization:.1%} utilized\n" \
                   f"**Knowledge Items**: {len(context.relevant_knowledge)}\n" \
                   f"**Total Context Size**: {context.total_size} characters"
        
        return ""

class ValidatorTemplate(BaseAgentTemplate):
    """Context template optimized for RIF-Validator"""
    
    def __init__(self):
        super().__init__(AgentType.VALIDATOR)
    
    def get_template_sections(self) -> List[TemplateSection]:
        return [
            TemplateSection(
                TemplateSection.CLAUDE_CAPABILITIES,
                "Claude Validation Tools",
                "validation_tools_format",
                ContentPriority.CRITICAL,
                max_chars=300
            ),
            TemplateSection(
                TemplateSection.QUALITY_GUIDELINES,
                "Quality Gates & Standards",
                "quality_gates_format",
                ContentPriority.CRITICAL,
                max_chars=800
            ),
            TemplateSection(
                TemplateSection.PERFORMANCE_TARGETS,
                "Performance Validation",
                "perf_validation_format",
                ContentPriority.HIGH,
                max_chars=400
            ),
            TemplateSection(
                TemplateSection.TASK_CONTEXT,
                "Validation Task",
                "validation_task_format",
                ContentPriority.HIGH,
                max_chars=400
            ),
            TemplateSection(
                TemplateSection.RELEVANT_KNOWLEDGE,
                "Quality Patterns & Error Detection",
                "quality_knowledge_format",
                ContentPriority.MEDIUM,
                max_chars=600
            ),
            TemplateSection(
                TemplateSection.SUCCESS_CRITERIA,
                "Validation Success Criteria",
                "validation_success_format",
                ContentPriority.MEDIUM,
                max_chars=300
            )
        ]
    
    def format_section_content(self, section: TemplateSection,
                              content: Any, context: AgentContext) -> str:
        if section.section_type == TemplateSection.CLAUDE_CAPABILITIES:
            claude_items = [item for item in context.relevant_knowledge 
                           if item.type == ContextType.CLAUDE_CODE_CAPABILITIES]
            if claude_items:
                return f"## 🔍 Validation Tools\n\n{claude_items[0].content}\n\n" \
                      f"**Validation Focus**: Use Bash for testing, Read for code review."
        
        elif section.section_type == TemplateSection.QUALITY_GUIDELINES:
            quality_items = [item for item in context.relevant_knowledge 
                           if item.type == ContextType.QUALITY_PATTERNS]
            error_items = [item for item in context.relevant_knowledge 
                          if item.type == ContextType.ERROR_PATTERNS]
            
            guidelines = []
            if quality_items:
                guidelines.append(f"**Quality Patterns**: {quality_items[0].content}")
            if error_items:
                guidelines.append(f"**Error Patterns**: {error_items[0].content}")
            
            base_guidelines = [
                "**Testing Requirements**: >80% coverage, passing unit/integration tests",
                "**Performance Gates**: Sub-200ms response times, load testing validation", 
                "**Security Standards**: No critical vulnerabilities, input validation",
                "**Code Quality**: Linting passed, proper error handling, documentation"
            ]
            
            all_guidelines = guidelines + base_guidelines
            return f"## ✅ Quality Gates & Standards\n\n" + "\n\n".join(all_guidelines)
        
        elif section.section_type == TemplateSection.PERFORMANCE_TARGETS:
            return f"## ⚡ Performance Validation Requirements\n\n" \
                   f"• **Response Time**: Must achieve <200ms average, <500ms P99\n" \
                   f"• **Load Testing**: Validate 10+ concurrent requests\n" \
                   f"• **Resource Usage**: Monitor memory, CPU, cache performance\n" \
                   f"• **Error Handling**: Test failure scenarios and recovery"
        
        elif section.section_type == TemplateSection.TASK_CONTEXT:
            task_desc = context.task_specific_context.get('description', 'No description provided')
            return f"## 🎯 Validation Task\n\n" \
                   f"**Implementation to Validate**: {task_desc}\n\n" \
                   f"**Validation Scope**: Functional correctness, performance compliance, quality standards"
        
        elif section.section_type == TemplateSection.RELEVANT_KNOWLEDGE:
            security_items = [item for item in context.relevant_knowledge 
                            if item.type == ContextType.SECURITY_PATTERNS]
            perf_items = [item for item in context.relevant_knowledge 
                         if item.type == ContextType.PERFORMANCE_DATA]
            
            knowledge_parts = []
            if security_items:
                knowledge_parts.append(f"**Security Patterns**: {security_items[0].content}")
            if perf_items:
                knowledge_parts.append(f"**Performance Data**: {perf_items[0].content}")
            
            if knowledge_parts:
                return f"## 📚 Quality Knowledge\n\n" + "\n\n".join(knowledge_parts)
        
        elif section.section_type == TemplateSection.SUCCESS_CRITERIA:
            return f"## ✅ Validation Success Criteria\n\n" \
                   f"• All tests passing with >80% coverage\n" \
                   f"• Performance targets met (sub-200ms)\n" \
                   f"• No critical security vulnerabilities\n" \
                   f"• Code quality standards satisfied\n" \
                   f"• Integration with existing systems validated"
        
        return ""

class ContextTemplateEngine:
    """Main engine for managing agent-specific context templates"""
    
    def __init__(self):
        self.templates = {
            AgentType.ANALYST: AnalystTemplate(),
            AgentType.IMPLEMENTER: ImplementerTemplate(), 
            AgentType.VALIDATOR: ValidatorTemplate(),
            # Add more templates as needed
        }
        
        # Generic template for agents without specific templates
        self.generic_template = BaseAgentTemplate(AgentType.IMPLEMENTER)
        self.formatting_metrics = []
    
    def render_context_for_agent(self, agent_context: AgentContext) -> str:
        """Render context using appropriate agent template"""
        template = self.templates.get(agent_context.agent_type, self.generic_template)
        
        start_time = time.time()
        formatted_context = template.render_context(agent_context)
        formatting_time = (time.time() - start_time) * 1000
        
        # Collect metrics
        if hasattr(template, 'formatting_metrics') and template.formatting_metrics:
            self.formatting_metrics.extend(template.formatting_metrics)
        
        return formatted_context
    
    def get_template_performance_stats(self) -> Dict[str, Any]:
        """Get template rendering performance statistics"""
        if not self.formatting_metrics:
            return {"status": "no_metrics"}
        
        # Group by agent type
        agent_stats = {}
        for metric in self.formatting_metrics:
            agent_type = metric.agent_type.value
            if agent_type not in agent_stats:
                agent_stats[agent_type] = []
            agent_stats[agent_type].append(metric)
        
        stats = {}
        for agent_type, metrics in agent_stats.items():
            formatting_times = [m.formatting_time_ms for m in metrics]
            output_sizes = [m.output_size_chars for m in metrics]
            truncation_rate = len([m for m in metrics if m.truncation_applied]) / len(metrics)
            
            stats[agent_type] = {
                "total_renders": len(metrics),
                "avg_formatting_time_ms": sum(formatting_times) / len(formatting_times),
                "p95_formatting_time_ms": sorted(formatting_times)[int(len(formatting_times) * 0.95)],
                "avg_output_size_chars": sum(output_sizes) / len(output_sizes),
                "truncation_rate": truncation_rate,
                "sub_50ms_compliance": len([t for t in formatting_times if t < 50]) / len(formatting_times)
            }
        
        return {
            "timestamp": datetime.now().isoformat(),
            "agent_stats": stats,
            "total_renders": len(self.formatting_metrics),
            "overall_avg_time_ms": sum(m.formatting_time_ms for m in self.formatting_metrics) / len(self.formatting_metrics)
        }
    
    def optimize_template_for_agent(self, agent_type: AgentType, 
                                   performance_data: Dict[str, Any]) -> Dict[str, Any]:
        """Optimize template based on performance data"""
        template = self.templates.get(agent_type)
        if not template:
            return {"status": "template_not_found"}
        
        # Analyze recent performance
        recent_metrics = [m for m in template.formatting_metrics if m.agent_type == agent_type][-10:]
        
        if not recent_metrics:
            return {"status": "insufficient_data"}
        
        avg_time = sum(m.formatting_time_ms for m in recent_metrics) / len(recent_metrics)
        truncation_rate = len([m for m in recent_metrics if m.truncation_applied]) / len(recent_metrics)
        
        optimizations = []
        
        # If formatting is slow, reduce context limits
        if avg_time > 100:  # 100ms threshold
            template.max_context_chars = int(template.max_context_chars * 0.9)
            optimizations.append(f"Reduced context limit to {template.max_context_chars} chars")
        
        # If truncation rate is high, adjust section priorities
        if truncation_rate > 0.3:  # 30% truncation rate
            # This would involve more complex template restructuring
            optimizations.append("High truncation rate detected - consider section rebalancing")
        
        return {
            "agent_type": agent_type.value,
            "optimizations_applied": optimizations,
            "new_context_limit": template.max_context_chars,
            "performance_improvement_expected": len(optimizations) > 0
        }

# CLI and Testing Interface
if __name__ == "__main__":
    import argparse
    ContextOptimizer = import_context_optimization_engine()['ContextOptimizer']
    
    parser = argparse.ArgumentParser(description="Agent Context Templates")
    parser.add_argument("--test", action="store_true", help="Test template rendering")
    parser.add_argument("--agent", type=str, choices=[a.value for a in AgentType],
                       help="Test specific agent type")
    parser.add_argument("--benchmark", action="store_true", help="Benchmark template performance")
    
    args = parser.parse_args()
    
    # Initialize components
    optimizer = ContextOptimizer()
    template_engine = ContextTemplateEngine()
    
    if args.test:
        print("=== Agent Context Templates Test ===\n")
        
        # Test data
        test_task = {
            "description": "Implement Context Optimization Engine with sub-200ms performance requirements",
            "complexity": "very_high", 
            "type": "implementation"
        }
        
        agents_to_test = [AgentType(args.agent)] if args.agent else list(template_engine.templates.keys())
        
        for agent_type in agents_to_test:
            print(f"Testing {agent_type.value} template...")
            
            # Generate context
            start_time = time.time()
            agent_context = optimizer.optimize_for_agent(agent_type, test_task, 123)
            context_time = (time.time() - start_time) * 1000
            
            # Render with template
            start_time = time.time() 
            formatted = template_engine.render_context_for_agent(agent_context)
            template_time = (time.time() - start_time) * 1000
            
            print(f"  Context generation: {context_time:.1f}ms")
            print(f"  Template rendering: {template_time:.1f}ms")
            print(f"  Output size: {len(formatted)} chars")
            print(f"  Template preview:")
            print("  " + "="*50)
            print("\n".join(["  " + line for line in formatted[:500].split('\n')]))
            if len(formatted) > 500:
                print("  ...")
            print("  " + "="*50 + "\n")
    
    elif args.benchmark:
        print("=== Template Performance Benchmark ===\n")
        
        test_task = {
            "description": "Benchmark template rendering performance for all agent types",
            "complexity": "high",
            "type": "performance_test"
        }
        
        # Run multiple iterations
        iterations = 20
        for agent_type in template_engine.templates.keys():
            print(f"Benchmarking {agent_type.value}...")
            
            times = []
            sizes = []
            
            for i in range(iterations):
                # Generate context
                agent_context = optimizer.optimize_for_agent(agent_type, test_task, 200 + i)
                
                # Measure template rendering
                start_time = time.time()
                formatted = template_engine.render_context_for_agent(agent_context)
                template_time = (time.time() - start_time) * 1000
                
                times.append(template_time)
                sizes.append(len(formatted))
            
            avg_time = sum(times) / len(times)
            p95_time = sorted(times)[int(len(times) * 0.95)]
            avg_size = sum(sizes) / len(sizes)
            sub_50ms_rate = len([t for t in times if t < 50]) / len(times)
            
            print(f"  Average: {avg_time:.1f}ms")
            print(f"  P95: {p95_time:.1f}ms") 
            print(f"  Sub-50ms: {sub_50ms_rate:.1%}")
            print(f"  Avg output size: {avg_size:.0f} chars")
            print()
        
        # Overall stats
        print("Overall Template Performance:")
        stats = template_engine.get_template_performance_stats()
        print(json.dumps(stats, indent=2, default=str))
    
    else:
        print("Agent Context Templates initialized")
        print("Available templates:")
        for agent_type in template_engine.templates.keys():
            print(f"  - {agent_type.value}")
        print("\nUse --test or --benchmark to test template functionality")