#!/usr/bin/env python3
"""
Frontend Specialist Agent - Issue #72
Specialized agent for frontend development, accessibility, and performance analysis.
"""

import json
import re
from datetime import datetime, timedelta
from typing import Dict, List, Any, Optional, Set, Tuple
from pathlib import Path
import logging

from domain_agent_base import DomainAgent, TaskResult, AgentStatus

logger = logging.getLogger(__name__)

class FrontendSpecialistAgent(DomainAgent):
    """Specialized agent for frontend development analysis and improvement"""
    
    def __init__(self):
        super().__init__(
            domain='frontend',
            capabilities=[
                'component_development',
                'state_management',
                'accessibility_testing',
                'responsive_design',
                'performance_optimization',
                'css_analysis',
                'javascript_analysis',
                'react_analysis',
                'vue_analysis'
            ]
        )
        
        # Frontend-specific patterns and rules
        self.accessibility_rules = self._load_accessibility_rules()
        self.performance_rules = self._load_performance_rules()
        self.best_practices = self._load_best_practices()
        
        logger.info("Frontend Specialist Agent initialized")
    
    def validate_capability(self, capability: str) -> bool:
        """Validate if this agent has a specific capability"""
        return capability in self.capabilities
    
    def execute_primary_task(self, task_data: Dict[str, Any]) -> TaskResult:
        """
        Execute the primary task for the Frontend Specialist Agent
        
        Args:
            task_data: Task data including component_code, context, and task_type
            
        Returns:
            TaskResult with execution details and analysis results
        """
        start_time = datetime.now()
        task_id = task_data.get('task_id', f'frontend_task_{start_time.strftime("%Y%m%d_%H%M%S")}')
        
        result = TaskResult(
            task_id=task_id,
            status=AgentStatus.RUNNING,
            start_time=start_time
        )
        
        try:
            task_type = task_data.get('task_type', 'analyze_component')
            component_code = task_data.get('component_code', '')
            context = task_data.get('context', {})
            
            if task_type == 'analyze_component':
                analysis_results = self.analyze_component(component_code, context)
                result.result_data = analysis_results
                result.confidence_score = analysis_results.get('confidence', 0.8)
                
            elif task_type == 'suggest_improvements':
                improvement_results = self.suggest_improvements(component_code)
                result.result_data = improvement_results
                result.confidence_score = 0.85
                
            elif task_type == 'accessibility_audit':
                component_info = self._identify_component_type(component_code, context)
                accessibility_issues = self.check_accessibility(component_code, component_info)
                result.result_data = {
                    'accessibility_issues': accessibility_issues,
                    'wcag_compliance_score': self._calculate_wcag_compliance_score(accessibility_issues)
                }
                result.confidence_score = 0.9
                
            elif task_type == 'performance_audit':
                component_info = self._identify_component_type(component_code, context)
                performance_issues = self.check_performance(component_code, component_info)
                result.result_data = {
                    'performance_issues': performance_issues,
                    'performance_score': self._calculate_performance_score(performance_issues)
                }
                result.confidence_score = 0.85
                
            else:
                raise ValueError(f'Unsupported task type: {task_type}')
            
            result.status = AgentStatus.COMPLETED
            
        except Exception as e:
            logger.error(f'Frontend specialist task execution failed: {e}')
            result.status = AgentStatus.FAILED
            result.error_message = str(e)
            result.confidence_score = 0.0
        
        finally:
            result.end_time = datetime.now()
            result.duration_seconds = (result.end_time - start_time).total_seconds()
        
        return result
    
    def analyze_component(self, component_code: str, context: Optional[Dict] = None) -> Dict[str, Any]:
        """
        Comprehensive frontend component analysis
        
        Args:
            component_code: Frontend component source code
            context: Optional context (file_path, framework, etc.)
            
        Returns:
            Analysis results with issues, metrics, and recommendations
        """
        analysis_start = datetime.now()
        
        # Determine component type and framework
        component_info = self._identify_component_type(component_code, context)
        
        issues = []
        metrics = {}
        
        # Core analysis areas
        issues.extend(self.check_accessibility(component_code, component_info))
        issues.extend(self.check_performance(component_code, component_info))
        issues.extend(self.check_best_practices(component_code, component_info))
        
        # Framework-specific analysis
        if component_info['framework'] == 'react':
            issues.extend(self._check_react_specific(component_code))
        elif component_info['framework'] == 'vue':
            issues.extend(self._check_vue_specific(component_code))
        
        # CSS analysis if present
        css_content = self._extract_css(component_code, component_info)
        if css_content:
            issues.extend(self._check_css_issues(css_content))
        
        # Calculate metrics
        metrics = self._calculate_component_metrics(component_code, component_info)
        
        analysis_duration = (datetime.now() - analysis_start).total_seconds()
        
        results = {
            'component_info': component_info,
            'issues': issues,
            'metrics': metrics,
            'analysis_duration': analysis_duration,
            'confidence': self._calculate_confidence_score(issues, metrics),
            'recommendations': self._generate_priority_recommendations(issues)
        }
        
        # Record this analysis
        self.record_analysis('component_analysis', results)
        
        return results
    
    def suggest_improvements(self, component_code: str, issues: Optional[List] = None) -> Dict[str, Any]:
        """
        Generate specific improvement suggestions
        
        Args:
            component_code: Source code to improve
            issues: Optional pre-identified issues
            
        Returns:
            Categorized improvement suggestions
        """
        if issues is None:
            analysis = self.analyze_component(component_code)
            issues = analysis['issues']
        
        suggestions = {
            'accessibility': self.suggest_a11y_improvements(component_code, issues),
            'performance': self.suggest_perf_improvements(component_code, issues),
            'patterns': self.suggest_pattern_improvements(component_code, issues),
            'code_quality': self.suggest_quality_improvements(component_code, issues)
        }
        
        # Add priority and effort estimates
        prioritized_suggestions = self._prioritize_suggestions(suggestions)
        
        return {
            'suggestions': suggestions,
            'prioritized': prioritized_suggestions,
            'implementation_order': self._suggest_implementation_order(prioritized_suggestions),
            'estimated_impact': self._estimate_improvement_impact(suggestions)
        }
    
    def check_accessibility(self, component_code: str, component_info: Dict) -> List[Dict]:
        """Check WCAG 2.1 AA compliance and accessibility best practices"""
        issues = []
        
        # Semantic HTML checks
        if not re.search(r'<(main|header|nav|section|article|aside|footer)', component_code, re.IGNORECASE):
            issues.append({
                'type': 'accessibility',
                'severity': 'medium',
                'category': 'semantic_html',
                'message': 'Missing semantic HTML elements (main, header, nav, etc.)',
                'wcag_reference': '1.3.1',
                'line': self._find_issue_line(component_code, 'semantic_html')
            })
        
        # Alt text for images
        img_without_alt = re.findall(r'<img(?![^>]*alt=)', component_code, re.IGNORECASE)
        if img_without_alt:
            issues.append({
                'type': 'accessibility',
                'severity': 'high',
                'category': 'alt_text',
                'message': f'Found {len(img_without_alt)} images without alt text',
                'wcag_reference': '1.1.1',
                'line': self._find_issue_line(component_code, '<img')
            })
        
        # Form labels
        inputs_without_labels = re.findall(r'<input(?![^>]*aria-label)(?![^>]*aria-labelledby)', component_code, re.IGNORECASE)
        if inputs_without_labels:
            issues.append({
                'type': 'accessibility',
                'severity': 'high',
                'category': 'form_labels',
                'message': f'Found {len(inputs_without_labels)} inputs without proper labels',
                'wcag_reference': '3.3.2',
                'line': self._find_issue_line(component_code, '<input')
            })
        
        # Color contrast (basic check for hardcoded colors)
        color_contrast_issues = self._check_color_contrast(component_code)
        issues.extend(color_contrast_issues)
        
        # Keyboard navigation
        if not re.search(r'onKeyDown|onKeyPress|tabIndex|role=', component_code, re.IGNORECASE):
            interactive_elements = re.findall(r'<(div|span)[^>]*onClick', component_code, re.IGNORECASE)
            if interactive_elements:
                issues.append({
                    'type': 'accessibility',
                    'severity': 'medium',
                    'category': 'keyboard_navigation',
                    'message': 'Interactive elements may not be keyboard accessible',
                    'wcag_reference': '2.1.1',
                    'line': self._find_issue_line(component_code, 'onClick')
                })
        
        return issues
    
    def check_performance(self, component_code: str, component_info: Dict) -> List[Dict]:
        """Check performance best practices and potential optimizations"""
        issues = []
        
        # React-specific performance checks
        if component_info['framework'] == 'react':
            # Unnecessary re-renders
            if 'useState' in component_code and not re.search(r'useMemo|useCallback|React\.memo', component_code):
                issues.append({
                    'type': 'performance',
                    'severity': 'medium',
                    'category': 'react_optimization',
                    'message': 'Consider using useMemo, useCallback, or React.memo to prevent unnecessary re-renders',
                    'line': self._find_issue_line(component_code, 'useState')
                })
            
            # Inline object/function creation in render
            inline_objects = re.findall(r'(?:onClick|onChange|style)=\{.*?(?:\{|\()', component_code)
            if inline_objects:
                issues.append({
                    'type': 'performance',
                    'severity': 'low',
                    'category': 'inline_creation',
                    'message': f'Found {len(inline_objects)} inline objects/functions that may cause re-renders',
                    'line': self._find_issue_line(component_code, 'onClick={')
                })
        
        # Large bundle size indicators
        import_count = len(re.findall(r'^import.*from', component_code, re.MULTILINE))
        if import_count > 15:
            issues.append({
                'type': 'performance',
                'severity': 'medium',
                'category': 'bundle_size',
                'message': f'High number of imports ({import_count}) may increase bundle size',
                'line': 1
            })
        
        # Heavy DOM operations
        query_selectors = re.findall(r'querySelector|getElementById|getElementsBy', component_code)
        if len(query_selectors) > 3:
            issues.append({
                'type': 'performance',
                'severity': 'medium',
                'category': 'dom_queries',
                'message': f'Multiple DOM queries ({len(query_selectors)}) detected - consider optimization',
                'line': self._find_issue_line(component_code, 'querySelector')
            })
        
        # Image optimization
        if re.search(r'<img[^>]*src=["\']/[^"\']*\.(jpg|jpeg|png|gif)', component_code, re.IGNORECASE):
            issues.append({
                'type': 'performance',
                'severity': 'low',
                'category': 'image_optimization',
                'message': 'Consider using optimized image formats (WebP) and lazy loading',
                'line': self._find_issue_line(component_code, '<img')
            })
        
        return issues
    
    def check_best_practices(self, component_code: str, component_info: Dict) -> List[Dict]:
        """Check general frontend best practices"""
        issues = []
        
        # Component size
        line_count = len(component_code.split('\n'))
        if line_count > 300:
            issues.append({
                'type': 'best_practice',
                'severity': 'medium',
                'category': 'component_size',
                'message': f'Component is large ({line_count} lines) - consider breaking into smaller components',
                'line': 1
            })
        
        # Hardcoded strings (i18n)
        hardcoded_strings = re.findall(r'>[^<>]*[a-zA-Z]{3,}[^<>]*<', component_code)
        if len(hardcoded_strings) > 5:
            issues.append({
                'type': 'best_practice',
                'severity': 'low',
                'category': 'internationalization',
                'message': f'Found {len(hardcoded_strings)} hardcoded text strings - consider i18n',
                'line': self._find_issue_line(component_code, '>')
            })
        
        # CSS-in-JS vs external styles
        inline_styles = len(re.findall(r'style=\{', component_code))
        if inline_styles > 5:
            issues.append({
                'type': 'best_practice',
                'severity': 'low',
                'category': 'styling',
                'message': f'High number of inline styles ({inline_styles}) - consider external CSS',
                'line': self._find_issue_line(component_code, 'style={')
            })
        
        # Error boundaries (React)
        if component_info['framework'] == 'react' and 'componentDidCatch' not in component_code and 'ErrorBoundary' not in component_code:
            if 'throw' in component_code or 'await' in component_code:
                issues.append({
                    'type': 'best_practice',
                    'severity': 'medium',
                    'category': 'error_handling',
                    'message': 'Consider adding error boundary for error handling',
                    'line': self._find_issue_line(component_code, 'throw')
                })
        
        return issues
    
    def suggest_a11y_improvements(self, component_code: str, issues: List) -> List[Dict]:
        """Generate accessibility improvement suggestions"""
        suggestions = []
        
        for issue in issues:
            if issue['type'] == 'accessibility':
                if issue['category'] == 'alt_text':
                    suggestions.append({
                        'type': 'accessibility',
                        'priority': 'high',
                        'effort': 'low',
                        'suggestion': 'Add descriptive alt text to all images',
                        'code_example': '<img src="..." alt="Descriptive text explaining the image" />',
                        'wcag_reference': issue.get('wcag_reference')
                    })
                elif issue['category'] == 'semantic_html':
                    suggestions.append({
                        'type': 'accessibility',
                        'priority': 'medium',
                        'effort': 'medium',
                        'suggestion': 'Replace generic divs with semantic HTML elements',
                        'code_example': '<main><section><article>Content</article></section></main>'
                    })
                elif issue['category'] == 'keyboard_navigation':
                    suggestions.append({
                        'type': 'accessibility',
                        'priority': 'high',
                        'effort': 'medium',
                        'suggestion': 'Add keyboard event handlers and proper tabIndex',
                        'code_example': '<div onClick={handler} onKeyDown={keyHandler} tabIndex={0} role="button">'
                    })
        
        return suggestions
    
    def suggest_perf_improvements(self, component_code: str, issues: List) -> List[Dict]:
        """Generate performance improvement suggestions"""
        suggestions = []
        
        for issue in issues:
            if issue['type'] == 'performance':
                if issue['category'] == 'react_optimization':
                    suggestions.append({
                        'type': 'performance',
                        'priority': 'high',
                        'effort': 'medium',
                        'suggestion': 'Implement React optimization techniques',
                        'code_example': 'const MemoizedComponent = React.memo(Component);\nconst memoizedValue = useMemo(() => expensiveCalculation, [deps]);'
                    })
                elif issue['category'] == 'image_optimization':
                    suggestions.append({
                        'type': 'performance',
                        'priority': 'medium',
                        'effort': 'low',
                        'suggestion': 'Use modern image formats and lazy loading',
                        'code_example': '<img src="image.webp" loading="lazy" alt="..." />'
                    })
        
        return suggestions
    
    def suggest_pattern_improvements(self, component_code: str, issues: List) -> List[Dict]:
        """Suggest architectural pattern improvements"""
        suggestions = []
        
        # Load domain-specific patterns
        patterns = self.load_domain_patterns()
        
        for issue in issues:
            if issue['category'] == 'component_size':
                suggestions.append({
                    'type': 'pattern',
                    'priority': 'medium',
                    'effort': 'high',
                    'suggestion': 'Apply component composition pattern',
                    'code_example': '// Break into smaller, focused components\nconst Header = () => <header>...</header>;\nconst Content = () => <main>...</main>;'
                })
        
        return suggestions
    
    def suggest_quality_improvements(self, component_code: str, issues: List) -> List[Dict]:
        """Suggest code quality improvements"""
        suggestions = []
        
        for issue in issues:
            if issue['type'] == 'best_practice':
                if issue['category'] == 'internationalization':
                    suggestions.append({
                        'type': 'quality',
                        'priority': 'low',
                        'effort': 'medium',
                        'suggestion': 'Implement internationalization (i18n)',
                        'code_example': 'const { t } = useTranslation();\n<span>{t("welcome_message")}</span>'
                    })
        
        return suggestions
    
    # Helper methods
    def _identify_component_type(self, code: str, context: Optional[Dict]) -> Dict[str, Any]:
        """Identify the component framework and type"""
        info = {
            'framework': 'unknown',
            'component_type': 'unknown',
            'has_state': False,
            'has_effects': False,
            'file_extension': context.get('file_path', '').split('.')[-1] if context else 'unknown'
        }
        
        # Framework detection
        if re.search(r'import.*react|from ["\']react', code, re.IGNORECASE):
            info['framework'] = 'react'
        elif re.search(r'import.*vue|from ["\']vue', code, re.IGNORECASE):
            info['framework'] = 'vue'
        elif re.search(r'<template>|<script>|<style>', code):
            info['framework'] = 'vue'
        
        # React specific
        if info['framework'] == 'react':
            info['has_state'] = bool(re.search(r'useState|useReducer', code))
            info['has_effects'] = bool(re.search(r'useEffect|useLayoutEffect', code))
            if re.search(r'class.*extends.*Component', code):
                info['component_type'] = 'class'
            elif re.search(r'const.*=.*\(.*\)\s*=>', code) or re.search(r'function.*\(', code):
                info['component_type'] = 'functional'
        
        return info
    
    def _extract_css(self, code: str, component_info: Dict) -> str:
        """Extract CSS content from component code"""
        css_content = ""
        
        # Extract CSS from style tags
        style_matches = re.findall(r'<style[^>]*>(.*?)</style>', code, re.DOTALL)
        css_content += '\n'.join(style_matches)
        
        # Extract CSS from style objects (CSS-in-JS)
        style_obj_matches = re.findall(r'style=\{([^}]+)\}', code)
        css_content += '\n'.join(style_obj_matches)
        
        return css_content.strip()
    
    def _check_css_issues(self, css_content: str) -> List[Dict]:
        """Check CSS-specific issues"""
        issues = []
        
        # Check for !important overuse
        important_count = len(re.findall(r'!important', css_content))
        if important_count > 3:
            issues.append({
                'type': 'css',
                'severity': 'medium',
                'category': 'specificity',
                'message': f'Overuse of !important ({important_count} instances)',
                'line': self._find_issue_line(css_content, '!important')
            })
        
        return issues
    
    def _check_color_contrast(self, code: str) -> List[Dict]:
        """Basic color contrast checking for hardcoded colors"""
        issues = []
        
        # Look for hardcoded color combinations that might have contrast issues
        color_patterns = re.findall(r'color:\s*[#\w]+|background-color:\s*[#\w]+', code, re.IGNORECASE)
        if len(color_patterns) > 0:
            issues.append({
                'type': 'accessibility',
                'severity': 'low',
                'category': 'color_contrast',
                'message': 'Manual verification needed for color contrast ratios',
                'wcag_reference': '1.4.3',
                'line': self._find_issue_line(code, 'color:')
            })
        
        return issues
    
    def _check_react_specific(self, code: str) -> List[Dict]:
        """React-specific checks"""
        issues = []
        
        # Check for missing keys in lists
        if re.search(r'\.map\s*\(', code) and not re.search(r'key\s*=', code):
            issues.append({
                'type': 'react',
                'severity': 'medium',
                'category': 'list_keys',
                'message': 'Missing keys in rendered lists',
                'line': self._find_issue_line(code, '.map(')
            })
        
        return issues
    
    def _check_vue_specific(self, code: str) -> List[Dict]:
        """Vue-specific checks"""
        issues = []
        
        # Vue template syntax checks
        if re.search(r'v-for\s*=', code) and not re.search(r':key\s*=', code):
            issues.append({
                'type': 'vue',
                'severity': 'medium',
                'category': 'list_keys',
                'message': 'Missing keys in v-for loops',
                'line': self._find_issue_line(code, 'v-for=')
            })
        
        return issues
    
    def _calculate_component_metrics(self, code: str, component_info: Dict) -> Dict[str, Any]:
        """Calculate component complexity and quality metrics"""
        lines = code.split('\n')
        
        return {
            'lines_of_code': len(lines),
            'cyclomatic_complexity': self._calculate_cyclomatic_complexity(code),
            'jsx_elements': len(re.findall(r'<[A-Z][^>]*>', code)),
            'hooks_count': len(re.findall(r'use[A-Z]\w*', code)),
            'prop_count': len(re.findall(r'props\.\w+', code)),
            'state_variables': len(re.findall(r'useState', code)),
            'comment_ratio': len([line for line in lines if line.strip().startswith('//')]) / max(len(lines), 1)
        }
    
    def _calculate_cyclomatic_complexity(self, code: str) -> int:
        """Simple cyclomatic complexity calculation"""
        complexity = 1  # Base complexity
        
        # Count decision points
        complexity += len(re.findall(r'\bif\b|\belse\b|\bwhile\b|\bfor\b|\bswitch\b|\bcase\b|\bcatch\b|\?\s*:', code))
        complexity += len(re.findall(r'&&|\|\|', code))
        
        return complexity
    
    def _calculate_confidence_score(self, issues: List, metrics: Dict) -> float:
        """Calculate confidence score for the analysis"""
        base_confidence = 0.85
        
        # Reduce confidence based on complexity
        complexity_penalty = min(metrics.get('cyclomatic_complexity', 0) * 0.02, 0.2)
        size_penalty = min(metrics.get('lines_of_code', 0) / 1000, 0.1)
        
        confidence = base_confidence - complexity_penalty - size_penalty
        return max(confidence, 0.5)
    
    def _generate_priority_recommendations(self, issues: List) -> List[Dict]:
        """Generate prioritized recommendations based on issues"""
        recommendations = []
        
        # Group issues by severity and type
        high_severity = [i for i in issues if i['severity'] == 'high']
        medium_severity = [i for i in issues if i['severity'] == 'medium']
        
        if high_severity:
            recommendations.append({
                'priority': 1,
                'title': 'Critical Issues',
                'description': f'Address {len(high_severity)} high-severity issues first',
                'issues': high_severity[:3]  # Top 3 high-severity issues
            })
        
        if medium_severity:
            recommendations.append({
                'priority': 2,
                'title': 'Important Improvements',
                'description': f'Address {len(medium_severity)} medium-severity issues',
                'issues': medium_severity[:5]  # Top 5 medium-severity issues
            })
        
        return recommendations
    
    def _prioritize_suggestions(self, suggestions: Dict[str, List]) -> List[Dict]:
        """Prioritize suggestions across all categories"""
        all_suggestions = []
        
        for category, category_suggestions in suggestions.items():
            for suggestion in category_suggestions:
                suggestion['category'] = category
                all_suggestions.append(suggestion)
        
        # Sort by priority (high first) then by effort (low first)
        priority_order = {'high': 3, 'medium': 2, 'low': 1}
        effort_order = {'low': 3, 'medium': 2, 'high': 1}
        
        return sorted(all_suggestions, key=lambda x: (
            priority_order.get(x.get('priority', 'low'), 0),
            effort_order.get(x.get('effort', 'high'), 0)
        ), reverse=True)
    
    def _suggest_implementation_order(self, prioritized_suggestions: List[Dict]) -> List[str]:
        """Suggest implementation order for suggestions"""
        return [
            f"{i+1}. {suggestion['suggestion'][:60]}..."
            for i, suggestion in enumerate(prioritized_suggestions[:5])
        ]
    
    def _estimate_improvement_impact(self, suggestions: Dict[str, List]) -> Dict[str, str]:
        """Estimate the impact of implementing suggestions"""
        impact = {}
        
        for category, category_suggestions in suggestions.items():
            high_priority_count = len([s for s in category_suggestions if s.get('priority') == 'high'])
            
            if high_priority_count >= 2:
                impact[category] = 'high'
            elif high_priority_count > 0:
                impact[category] = 'medium'
            else:
                impact[category] = 'low'
        
        return impact
    
    def _find_issue_line(self, code: str, pattern: str) -> int:
        """Find the line number where an issue occurs"""
        lines = code.split('\n')
        for i, line in enumerate(lines):
            if pattern.lower() in line.lower():
                return i + 1
        return 1
    
    def _calculate_wcag_compliance_score(self, accessibility_issues: List[Dict]) -> float:
        """Calculate WCAG compliance score based on accessibility issues"""
        if not accessibility_issues:
            return 1.0
        
        # Weight issues by severity
        severity_weights = {'high': 0.4, 'medium': 0.2, 'low': 0.1}
        total_penalty = sum(severity_weights.get(issue.get('severity', 'medium'), 0.2) for issue in accessibility_issues)
        
        # Cap at 0.0 minimum
        return max(0.0, 1.0 - total_penalty)
    
    def _calculate_performance_score(self, performance_issues: List[Dict]) -> float:
        """Calculate performance score based on performance issues"""
        if not performance_issues:
            return 1.0
        
        # Weight issues by severity
        severity_weights = {'high': 0.3, 'medium': 0.15, 'low': 0.05}
        total_penalty = sum(severity_weights.get(issue.get('severity', 'medium'), 0.15) for issue in performance_issues)
        
        # Cap at 0.0 minimum
        return max(0.0, 1.0 - total_penalty)
    
    def _load_accessibility_rules(self) -> Dict[str, Any]:
        """Load accessibility rules and guidelines"""
        return {
            'wcag_2_1_aa': {
                '1.1.1': 'All images must have alt text',
                '1.3.1': 'Use semantic HTML elements',
                '1.4.3': 'Color contrast ratio must be at least 4.5:1',
                '2.1.1': 'All functionality must be keyboard accessible',
                '3.3.2': 'Form inputs must have proper labels'
            }
        }
    
    def _load_performance_rules(self) -> Dict[str, Any]:
        """Load performance optimization rules"""
        return {
            'react': {
                'memoization': 'Use React.memo, useMemo, useCallback appropriately',
                'bundle_size': 'Keep bundle size under reasonable limits',
                'dom_queries': 'Minimize DOM query operations'
            },
            'general': {
                'images': 'Use optimized image formats and lazy loading',
                'css': 'Avoid excessive inline styles'
            }
        }
    
    def _load_best_practices(self) -> Dict[str, Any]:
        """Load general frontend best practices"""
        return {
            'component_design': {
                'single_responsibility': 'Each component should have a single responsibility',
                'size_limit': 'Keep components under 300 lines when possible',
                'composition': 'Prefer composition over inheritance'
            },
            'code_quality': {
                'error_handling': 'Implement proper error boundaries',
                'internationalization': 'Plan for i18n from the start',
                'testing': 'Write unit tests for components'
            }
        }