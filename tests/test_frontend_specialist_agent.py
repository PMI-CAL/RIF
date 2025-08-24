#!/usr/bin/env python3
"""
Comprehensive tests for Frontend Specialist Agent - Issue #72
"""

import json
import pytest
import sys
from pathlib import Path
from unittest.mock import patch, mock_open

# Add the agents directory to the Python path
sys.path.insert(0, str(Path(__file__).parent.parent / 'claude' / 'agents'))

from frontend_specialist_agent import FrontendSpecialistAgent
from domain_agent_base import DomainAgent

class TestFrontendSpecialistAgent:
    """Test suite for the Frontend Specialist Agent"""
    
    @pytest.fixture
    def agent(self):
        """Create a frontend specialist agent instance for testing"""
        return FrontendSpecialistAgent()
    
    @pytest.fixture
    def sample_react_component(self):
        """Sample React component with various issues for testing"""
        return '''
import React, { useState } from 'react';

const UserProfile = () => {
    const [user, setUser] = useState(null);
    
    return (
        <div onClick={() => console.log('clicked')} style={{ color: '#fff', backgroundColor: '#000' }}>
            <img src="/profile.jpg" />
            <input type="email" placeholder="Enter email" />
            <div style={{ fontSize: '12px' }}>
                {users.map(user => (
                    <div>{user.name}</div>
                ))}
            </div>
            <span>Welcome to our app</span>
            <button>Submit</button>
        </div>
    );
};

export default UserProfile;
'''
    
    @pytest.fixture
    def sample_vue_component(self):
        """Sample Vue component for testing"""
        return '''
<template>
    <div>
        <img :src="userImage" />
        <input v-model="email" type="email" />
        <div v-for="user in users">
            {{ user.name }}
        </div>
    </div>
</template>

<script>
export default {
    data() {
        return {
            email: '',
            users: []
        }
    }
}
</script>
'''
    
    def test_agent_initialization(self, agent):
        """Test that the agent initializes correctly"""
        assert agent.domain == 'frontend'
        assert 'component_development' in agent.capabilities
        assert 'accessibility_testing' in agent.capabilities
        assert 'responsive_design' in agent.capabilities
        assert agent.agent_id.startswith('frontend_agent_')
    
    def test_capability_validation(self, agent):
        """Test capability validation"""
        assert agent.validate_capability('component_development') is True
        assert agent.validate_capability('accessibility_testing') is True
        assert agent.validate_capability('nonexistent_capability') is False
    
    def test_get_domain_info(self, agent):
        """Test domain information retrieval"""
        info = agent.get_domain_info()
        assert info['domain'] == 'frontend'
        assert 'capabilities' in info
        assert 'created_at' in info
        assert info['analyses_performed'] == 0
    
    def test_analyze_react_component(self, agent, sample_react_component):
        """Test analysis of React component"""
        analysis = agent.analyze_component(sample_react_component, {'file_path': 'UserProfile.jsx'})
        
        # Check analysis structure
        assert 'component_info' in analysis
        assert 'issues' in analysis
        assert 'metrics' in analysis
        assert 'confidence' in analysis
        
        # Check component identification
        assert analysis['component_info']['framework'] == 'react'
        assert analysis['component_info']['has_state'] is True
        
        # Check that issues were found
        assert len(analysis['issues']) > 0
        
        # Check for specific accessibility issues
        accessibility_issues = [i for i in analysis['issues'] if i['type'] == 'accessibility']
        assert len(accessibility_issues) > 0
        
        # Should find alt text issue
        alt_text_issues = [i for i in accessibility_issues if i['category'] == 'alt_text']
        assert len(alt_text_issues) > 0
    
    def test_analyze_vue_component(self, agent, sample_vue_component):
        """Test analysis of Vue component"""
        analysis = agent.analyze_component(sample_vue_component, {'file_path': 'UserProfile.vue'})
        
        # Check framework identification
        assert analysis['component_info']['framework'] == 'vue'
        
        # Check for Vue-specific issues
        vue_issues = [i for i in analysis['issues'] if i['type'] == 'vue']
        list_key_issues = [i for i in vue_issues if i['category'] == 'list_keys']
        assert len(list_key_issues) > 0  # Should find missing :key in v-for
    
    def test_accessibility_checks(self, agent):
        """Test accessibility checking functionality"""
        # Component with accessibility issues
        bad_component = '''
        <div onClick={handleClick}>
            <img src="test.jpg" />
            <input type="text" />
        </div>
        '''
        
        issues = agent.check_accessibility(bad_component, {'framework': 'react'})
        
        # Should find multiple accessibility issues
        assert len(issues) >= 2
        
        # Should find alt text issue
        alt_issues = [i for i in issues if i['category'] == 'alt_text']
        assert len(alt_issues) > 0
        
        # Should find form label issue
        label_issues = [i for i in issues if i['category'] == 'form_labels']
        assert len(label_issues) > 0
        
        # Should find keyboard navigation issue
        keyboard_issues = [i for i in issues if i['category'] == 'keyboard_navigation']
        assert len(keyboard_issues) > 0
    
    def test_performance_checks(self, agent):
        """Test performance checking functionality"""
        # Component with performance issues
        perf_component = '''
        import React, { useState } from 'react';
        import Module1 from './module1';
        import Module2 from './module2';
        // ... many more imports (simulate 20+ imports)
        
        const Component = () => {
            const [state, setState] = useState(0);
            
            return (
                <div>
                    <button onClick={() => console.log('click')}>
                        Click me
                    </button>
                    <img src="/large-image.jpg" />
                </div>
            );
        };
        '''
        
        issues = agent.check_performance(perf_component, {'framework': 'react'})
        
        # Should find performance issues
        assert len(issues) > 0
        
        # Should suggest React optimization
        react_opt_issues = [i for i in issues if i['category'] == 'react_optimization']
        assert len(react_opt_issues) > 0
    
    def test_best_practices_checks(self, agent):
        """Test best practices checking"""
        # Large component with hardcoded strings
        large_component = '''
        const LargeComponent = () => {
            return (
                <div>
                    <h1>Welcome to our application</h1>
                    <p>This is some hardcoded text</p>
                    <span>Another hardcoded string</span>
                    <div>More text here</div>
                    <p>Even more hardcoded content</p>
                    <span>Additional text content</span>
                    ''' + '\n'.join([f'<div>Line {i}</div>' for i in range(300)]) + '''
                </div>
            );
        };
        '''
        
        issues = agent.check_best_practices(large_component, {'framework': 'react'})
        
        # Should find component size issue
        size_issues = [i for i in issues if i['category'] == 'component_size']
        assert len(size_issues) > 0
        
        # Should find i18n issue
        i18n_issues = [i for i in issues if i['category'] == 'internationalization']
        assert len(i18n_issues) > 0
    
    def test_suggest_improvements(self, agent, sample_react_component):
        """Test improvement suggestion generation"""
        suggestions = agent.suggest_improvements(sample_react_component)
        
        # Check structure
        assert 'suggestions' in suggestions
        assert 'prioritized' in suggestions
        assert 'implementation_order' in suggestions
        assert 'estimated_impact' in suggestions
        
        # Check suggestion categories
        assert 'accessibility' in suggestions['suggestions']
        assert 'performance' in suggestions['suggestions']
        assert 'patterns' in suggestions['suggestions']
        assert 'code_quality' in suggestions['suggestions']
        
        # Should have prioritized suggestions
        assert len(suggestions['prioritized']) > 0
        
        # Should have implementation order
        assert len(suggestions['implementation_order']) > 0
    
    def test_accessibility_suggestions(self, agent):
        """Test accessibility-specific suggestions"""
        issues = [
            {
                'type': 'accessibility',
                'category': 'alt_text',
                'severity': 'high',
                'wcag_reference': '1.1.1'
            },
            {
                'type': 'accessibility',
                'category': 'semantic_html',
                'severity': 'medium'
            }
        ]
        
        suggestions = agent.suggest_a11y_improvements('', issues)
        
        assert len(suggestions) == 2
        
        # Check alt text suggestion
        alt_suggestion = next((s for s in suggestions if 'alt text' in s['suggestion']), None)
        assert alt_suggestion is not None
        assert alt_suggestion['priority'] == 'high'
        assert 'code_example' in alt_suggestion
    
    def test_performance_suggestions(self, agent):
        """Test performance-specific suggestions"""
        issues = [
            {
                'type': 'performance',
                'category': 'react_optimization',
                'severity': 'medium'
            },
            {
                'type': 'performance',
                'category': 'image_optimization',
                'severity': 'low'
            }
        ]
        
        suggestions = agent.suggest_perf_improvements('', issues)
        
        assert len(suggestions) == 2
        
        # Check React optimization suggestion
        react_suggestion = next((s for s in suggestions if 'React' in s['suggestion']), None)
        assert react_suggestion is not None
        assert 'useMemo' in react_suggestion['code_example']
    
    def test_component_type_identification(self, agent):
        """Test component type and framework identification"""
        # React functional component
        react_func = '''
        import React from 'react';
        const Component = () => <div>Hello</div>;
        '''
        
        info = agent._identify_component_type(react_func, {})
        assert info['framework'] == 'react'
        assert info['component_type'] == 'functional'  # Should detect functional component
        
        # Vue component
        vue_comp = '''
        <template>
            <div>Hello</div>
        </template>
        '''
        
        info = agent._identify_component_type(vue_comp, {})
        assert info['framework'] == 'vue'
    
    def test_css_extraction_and_analysis(self, agent):
        """Test CSS extraction and analysis"""
        component_with_css = '''
        const Component = () => (
            <div>
                <style>
                    .test { color: red !important; background: blue !important; font-size: 12px !important; margin: 10px !important; }
                </style>
                <div style={{color: 'white', backgroundColor: 'black'}}>Content</div>
            </div>
        );
        '''
        
        css_content = agent._extract_css(component_with_css, {'framework': 'react'})
        assert '!important' in css_content
        
        css_issues = agent._check_css_issues(css_content)
        
        # Should find overuse of !important
        important_issues = [i for i in css_issues if i['category'] == 'specificity']
        assert len(important_issues) > 0
    
    def test_cyclomatic_complexity_calculation(self, agent):
        """Test cyclomatic complexity calculation"""
        simple_component = 'const Component = () => <div>Simple</div>;'
        complex_component = '''
        const Component = () => {
            if (condition1) {
                if (condition2) {
                    return <div>Complex</div>;
                } else if (condition3 && condition4) {
                    return <div>Very Complex</div>;
                }
            }
            for (let i = 0; i < 10; i++) {
                console.log(i);
            }
            return <div>Default</div>;
        };
        '''
        
        simple_complexity = agent._calculate_cyclomatic_complexity(simple_component)
        complex_complexity = agent._calculate_cyclomatic_complexity(complex_component)
        
        assert simple_complexity == 1
        assert complex_complexity > simple_complexity
    
    def test_metrics_calculation(self, agent, sample_react_component):
        """Test component metrics calculation"""
        component_info = {'framework': 'react'}
        metrics = agent._calculate_component_metrics(sample_react_component, component_info)
        
        assert 'lines_of_code' in metrics
        assert 'cyclomatic_complexity' in metrics
        assert 'hooks_count' in metrics
        assert 'state_variables' in metrics
        assert 'comment_ratio' in metrics
        
        assert metrics['lines_of_code'] > 0
        assert metrics['hooks_count'] > 0  # Should find useState
        assert metrics['state_variables'] > 0  # Should find useState
    
    def test_confidence_score_calculation(self, agent):
        """Test confidence score calculation"""
        # Simple case
        simple_issues = []
        simple_metrics = {'cyclomatic_complexity': 2, 'lines_of_code': 50}
        simple_confidence = agent._calculate_confidence_score(simple_issues, simple_metrics)
        
        # Complex case
        complex_metrics = {'cyclomatic_complexity': 20, 'lines_of_code': 1000}
        complex_confidence = agent._calculate_confidence_score([], complex_metrics)
        
        assert 0.5 <= simple_confidence <= 1.0
        assert 0.5 <= complex_confidence <= 1.0
        assert simple_confidence > complex_confidence
    
    def test_analysis_recording(self, agent, sample_react_component):
        """Test that analyses are recorded in history"""
        initial_count = len(agent.analysis_history)
        
        agent.analyze_component(sample_react_component)
        
        assert len(agent.analysis_history) == initial_count + 1
        
        # Check recorded analysis structure
        latest_analysis = agent.analysis_history[-1]
        assert 'timestamp' in latest_analysis
        assert 'analysis_type' in latest_analysis
        assert 'results_summary' in latest_analysis
    
    @patch('builtins.open', mock_open(read_data='{"pattern1": {"description": "test pattern"}}'))
    @patch('pathlib.Path.exists', return_value=True)
    def test_load_domain_patterns(self, mock_exists, agent):
        """Test loading domain-specific patterns"""
        patterns = agent.load_domain_patterns()
        assert isinstance(patterns, dict)
    
    @patch('builtins.open', mock_open())
    @patch('pathlib.Path.mkdir')
    def test_save_domain_learning(self, mock_mkdir, agent):
        """Test saving learning data"""
        learning_data = {
            'pattern': 'test_pattern',
            'effectiveness': 0.8,
            'context': 'React component optimization'
        }
        
        # Should not raise exception
        agent.save_domain_learning(learning_data)
    
    def test_performance_metrics(self, agent, sample_react_component):
        """Test agent performance metrics"""
        # Perform some analyses to generate history
        for _ in range(3):
            agent.analyze_component(sample_react_component)
        
        metrics = agent.get_performance_metrics()
        
        assert 'total_analyses' in metrics
        assert 'recent_analyses' in metrics
        assert 'avg_issues_per_analysis' in metrics
        assert 'avg_suggestions_per_analysis' in metrics
        
        assert metrics['total_analyses'] == 3
    
    def test_error_handling(self, agent):
        """Test error handling with malformed input"""
        # Empty component
        analysis = agent.analyze_component('')
        assert 'issues' in analysis
        
        # Malformed component
        malformed = '<div><img><input></div'  # Unclosed tags
        analysis = agent.analyze_component(malformed)
        assert 'issues' in analysis
        
        # Very large component (should still work)
        large_component = '<div>' + 'x' * 10000 + '</div>'
        analysis = agent.analyze_component(large_component)
        assert 'issues' in analysis
    
    def test_framework_specific_analysis(self, agent):
        """Test framework-specific analysis features"""
        # React component with missing keys
        react_with_list = '''
        const Component = () => (
            <div>
                {items.map(item => (
                    <div>{item.name}</div>
                ))}
            </div>
        );
        '''
        
        react_issues = agent._check_react_specific(react_with_list)
        key_issues = [i for i in react_issues if i['category'] == 'list_keys']
        assert len(key_issues) > 0
        
        # Vue component with missing keys
        vue_with_loop = '''
        <template>
            <div>
                <div v-for="item in items">{{ item.name }}</div>
            </div>
        </template>
        '''
        
        vue_issues = agent._check_vue_specific(vue_with_loop)
        key_issues = [i for i in vue_issues if i['category'] == 'list_keys']
        assert len(key_issues) > 0
    
    def test_suggestion_prioritization(self, agent):
        """Test suggestion prioritization logic"""
        suggestions = {
            'accessibility': [
                {'priority': 'high', 'effort': 'low', 'suggestion': 'Fix alt text'},
                {'priority': 'medium', 'effort': 'high', 'suggestion': 'Improve semantics'}
            ],
            'performance': [
                {'priority': 'low', 'effort': 'low', 'suggestion': 'Optimize images'},
                {'priority': 'high', 'effort': 'medium', 'suggestion': 'Add memoization'}
            ]
        }
        
        prioritized = agent._prioritize_suggestions(suggestions)
        
        # Should prioritize high priority, low effort items first
        assert len(prioritized) == 4
        assert prioritized[0]['priority'] == 'high'
        assert prioritized[0]['effort'] == 'low'
    
    def test_implementation_order_suggestion(self, agent):
        """Test implementation order suggestion"""
        prioritized_suggestions = [
            {'suggestion': 'Fix critical accessibility issues first'},
            {'suggestion': 'Add performance optimizations'},
            {'suggestion': 'Improve code structure'},
            {'suggestion': 'Add internationalization'},
            {'suggestion': 'Enhance error handling'}
        ]
        
        order = agent._suggest_implementation_order(prioritized_suggestions)
        
        assert len(order) == 5
        assert order[0].startswith('1.')
        assert 'Fix critical accessibility' in order[0]
    
    def test_improvement_impact_estimation(self, agent):
        """Test improvement impact estimation"""
        suggestions = {
            'accessibility': [
                {'priority': 'high'},
                {'priority': 'high'},
                {'priority': 'medium'}
            ],
            'performance': [
                {'priority': 'medium'}
            ],
            'quality': []
        }
        
        impact = agent._estimate_improvement_impact(suggestions)
        
        assert impact['accessibility'] == 'high'  # 2+ high priority
        assert impact['performance'] == 'low'     # No high priority, only medium
        assert impact['quality'] == 'low'          # No high priority
    
    def test_wcag_compliance_checking(self, agent):
        """Test WCAG compliance checking"""
        # Component with WCAG violations
        non_compliant = '''
        <div>
            <img src="logo.png">
            <input type="password">
            <div onclick="handleClick()" style="color: #999; background: #ccc;">
                Click me
            </div>
        </div>
        '''
        
        issues = agent.check_accessibility(non_compliant, {'framework': 'html'})
        
        # Should find WCAG violations with references
        wcag_issues = [i for i in issues if 'wcag_reference' in i]
        assert len(wcag_issues) > 0
        
        # Should reference specific WCAG guidelines
        alt_text_issue = next((i for i in wcag_issues if i['wcag_reference'] == '1.1.1'), None)
        assert alt_text_issue is not None


if __name__ == '__main__':
    # Run tests if executed directly
    pytest.main([__file__, '-v'])