#!/usr/bin/env python3
"""
Frontend Specialist Agent Demonstration - Issue #72
Shows how to use the frontend specialist agent for component analysis.
"""

import json
import sys
from pathlib import Path

# Add the agents directory to the Python path
sys.path.insert(0, str(Path(__file__).parent / 'claude' / 'agents'))

from frontend_specialist_agent import FrontendSpecialistAgent

def demo_react_analysis():
    """Demonstrate React component analysis"""
    print("🔍 Frontend Specialist Agent - React Component Analysis Demo")
    print("=" * 60)
    
    # Sample React component with various issues
    react_component = '''
import React, { useState } from 'react';

const UserProfile = ({ user }) => {
    const [isEditing, setIsEditing] = useState(false);
    const [formData, setFormData] = useState(user);
    
    const handleSubmit = () => {
        // Submit logic here
        console.log('Submitting:', formData);
    };
    
    return (
        <div onClick={() => setIsEditing(true)} style={{ color: '#666', backgroundColor: '#f0f0f0' }}>
            <img src={user.avatar} />
            <h2>Welcome back!</h2>
            
            {isEditing ? (
                <form onSubmit={handleSubmit}>
                    <input 
                        type="email" 
                        value={formData.email}
                        placeholder="Enter your email"
                        onChange={e => setFormData({...formData, email: e.target.value})}
                    />
                    <input 
                        type="password"
                        value={formData.password} 
                        placeholder="Password"
                        onChange={e => setFormData({...formData, password: e.target.value})}
                    />
                    <button type="submit">Update Profile</button>
                </form>
            ) : (
                <div>
                    <p>Email: {user.email}</p>
                    <div style={{ fontSize: '14px', color: '#999' }}>
                        {user.notifications.map(notification => (
                            <div>{notification.message}</div>
                        ))}
                    </div>
                </div>
            )}
        </div>
    );
};

export default UserProfile;
'''
    
    # Create agent and analyze
    agent = FrontendSpecialistAgent()
    
    print("📝 Analyzing React component...")
    analysis = agent.analyze_component(
        react_component, 
        context={'file_path': 'UserProfile.jsx', 'framework': 'react'}
    )
    
    print(f"\n📊 Analysis Results:")
    print(f"Framework detected: {analysis['component_info']['framework']}")
    print(f"Component type: {analysis['component_info']['component_type']}")
    print(f"Has state: {analysis['component_info']['has_state']}")
    print(f"Issues found: {len(analysis['issues'])}")
    print(f"Confidence score: {analysis['confidence']:.2f}")
    print(f"Analysis time: {analysis['analysis_duration']:.3f}s")
    
    print(f"\n🚨 Issues Found:")
    for i, issue in enumerate(analysis['issues'][:5], 1):  # Show first 5 issues
        severity_icon = {"high": "🔴", "medium": "🟡", "low": "🟢"}.get(issue['severity'], "⚪")
        print(f"{i}. {severity_icon} [{issue['type'].upper()}] {issue['message']}")
        if 'wcag_reference' in issue:
            print(f"   📋 WCAG: {issue['wcag_reference']}")
    
    if len(analysis['issues']) > 5:
        print(f"   ... and {len(analysis['issues']) - 5} more issues")
    
    return analysis

def demo_suggestions():
    """Demonstrate improvement suggestions"""
    print("\n" + "=" * 60)
    print("💡 Improvement Suggestions Demo")
    print("=" * 60)
    
    agent = FrontendSpecialistAgent()
    
    # Vue component with issues for variety
    vue_component = '''
<template>
    <div class="user-dashboard">
        <img :src="userAvatar" />
        <h1>User Dashboard</h1>
        
        <div v-for="item in dashboardItems" class="dashboard-item">
            <span>{{ item.title }}</span>
            <input v-model="item.value" type="text" />
        </div>
        
        <div @click="handleAction()" :style="{ color: lightGray, background: darkBlue }">
            Click me for action
        </div>
        
        <button @click="submitData()">Submit All Data</button>
    </div>
</template>

<script>
export default {
    data() {
        return {
            userAvatar: '/default-avatar.png',
            dashboardItems: [],
            lightGray: '#ccc',
            darkBlue: '#003366'
        }
    },
    methods: {
        handleAction() {
            console.log('Action triggered');
        },
        submitData() {
            // Submit logic
        }
    }
}
</script>

<style>
.user-dashboard {
    padding: 20px;
    color: #333 !important;
    background: #fff !important;
    font-size: 16px !important;
}
.dashboard-item {
    margin: 10px 0 !important;
}
</style>
'''
    
    print("📝 Analyzing Vue component and generating suggestions...")
    suggestions = agent.suggest_improvements(vue_component)
    
    print(f"\n💡 Improvement Suggestions:")
    
    for category, category_suggestions in suggestions['suggestions'].items():
        if category_suggestions:
            print(f"\n📂 {category.title()}:")
            for i, suggestion in enumerate(category_suggestions[:3], 1):
                priority_icon = {"high": "🔴", "medium": "🟡", "low": "🟢"}.get(suggestion.get('priority'), "⚪")
                effort_text = suggestion.get('effort', 'unknown').upper()
                print(f"  {i}. {priority_icon} [{effort_text} EFFORT] {suggestion['suggestion']}")
                if 'code_example' in suggestion:
                    print(f"     💻 Example: {suggestion['code_example'][:80]}...")
    
    print(f"\n🎯 Implementation Priority Order:")
    for i, step in enumerate(suggestions['implementation_order'][:5], 1):
        print(f"  {i}. {step}")
    
    print(f"\n📈 Estimated Impact:")
    for category, impact in suggestions['estimated_impact'].items():
        impact_icon = {"high": "🚀", "medium": "📈", "low": "📊"}.get(impact, "📋")
        print(f"  {category.title()}: {impact_icon} {impact.upper()}")

def demo_accessibility_focus():
    """Demonstrate accessibility-focused analysis"""
    print("\n" + "=" * 60)
    print("♿ Accessibility Analysis Demo")
    print("=" * 60)
    
    agent = FrontendSpecialistAgent()
    
    # Component with accessibility issues
    accessibility_component = '''
<div class="login-form">
    <h1>Login</h1>
    
    <img src="/company-logo.png" />
    
    <div class="form-group">
        <input type="email" placeholder="Email" />
    </div>
    
    <div class="form-group">
        <input type="password" placeholder="Password" />
    </div>
    
    <div class="custom-button" onClick="handleLogin()" style="color: #aaa; background: #ddd;">
        Login
    </div>
    
    <div class="links">
        <span onClick="forgotPassword()" style="color: #999;">Forgot Password?</span>
        <span onClick="createAccount()" style="color: #888;">Create Account</span>
    </div>
</div>
'''
    
    print("📝 Analyzing accessibility issues...")
    
    # Get just accessibility issues
    component_info = agent._identify_component_type(accessibility_component, {})
    accessibility_issues = agent.check_accessibility(accessibility_component, component_info)
    
    print(f"\n♿ Accessibility Issues Found: {len(accessibility_issues)}")
    
    for i, issue in enumerate(accessibility_issues, 1):
        wcag_ref = f" (WCAG {issue['wcag_reference']})" if 'wcag_reference' in issue else ""
        severity_icon = {"high": "🔴", "medium": "🟡", "low": "🟢"}.get(issue['severity'], "⚪")
        print(f"{i}. {severity_icon} {issue['message']}{wcag_ref}")
        print(f"   📍 Category: {issue['category']}")
    
    # Get accessibility-specific suggestions
    a11y_suggestions = agent.suggest_a11y_improvements(accessibility_component, accessibility_issues)
    
    print(f"\n🔧 Accessibility Improvement Suggestions:")
    for i, suggestion in enumerate(a11y_suggestions, 1):
        priority_icon = {"high": "🔴", "medium": "🟡", "low": "🟢"}.get(suggestion.get('priority'), "⚪")
        print(f"{i}. {priority_icon} {suggestion['suggestion']}")
        if 'code_example' in suggestion:
            print(f"   💻 {suggestion['code_example']}")
        if 'wcag_reference' in suggestion:
            print(f"   📋 WCAG: {suggestion['wcag_reference']}")

def demo_performance_analysis():
    """Demonstrate performance analysis"""
    print("\n" + "=" * 60)
    print("⚡ Performance Analysis Demo") 
    print("=" * 60)
    
    agent = FrontendSpecialistAgent()
    
    # React component with performance issues
    performance_component = '''
import React, { useState, useEffect } from 'react';
import LargeLibrary from 'large-library';
import AnotherLibrary from 'another-library';
import ThirdLibrary from 'third-library';
// ... 15+ more imports

const ExpensiveComponent = ({ data, onUpdate }) => {
    const [items, setItems] = useState([]);
    const [loading, setLoading] = useState(false);
    
    useEffect(() => {
        // Expensive operation
        const processedData = data.map(item => ({
            ...item,
            processed: true,
            timestamp: Date.now()
        }));
        setItems(processedData);
    }, [data]);
    
    return (
        <div>
            {items.map(item => (
                <div 
                    key={item.id}
                    onClick={() => onUpdate(item)}
                    style={{
                        background: loading ? '#f0f0f0' : '#ffffff',
                        border: '1px solid #ccc',
                        margin: '5px'
                    }}
                >
                    <img src={item.largeImage} alt="Item image" />
                    <span>{item.title}</span>
                    <button onClick={() => document.getElementById('modal').style.display = 'block'}>
                        View Details
                    </button>
                </div>
            ))}
        </div>
    );
};
'''
    
    print("📝 Analyzing performance patterns...")
    
    component_info = agent._identify_component_type(performance_component, {'framework': 'react'})
    performance_issues = agent.check_performance(performance_component, component_info)
    
    print(f"\n⚡ Performance Issues Found: {len(performance_issues)}")
    
    for i, issue in enumerate(performance_issues, 1):
        severity_icon = {"high": "🔴", "medium": "🟡", "low": "🟢"}.get(issue['severity'], "⚪")
        print(f"{i}. {severity_icon} [{issue['category'].upper()}] {issue['message']}")
    
    # Get performance suggestions
    perf_suggestions = agent.suggest_perf_improvements(performance_component, performance_issues)
    
    print(f"\n🚀 Performance Improvement Suggestions:")
    for i, suggestion in enumerate(perf_suggestions, 1):
        priority_icon = {"high": "🔴", "medium": "🟡", "low": "🟢"}.get(suggestion.get('priority'), "⚪")
        print(f"{i}. {priority_icon} {suggestion['suggestion']}")
        if 'code_example' in suggestion:
            print(f"   💻 {suggestion['code_example'][:100]}...")

def main():
    """Run the frontend specialist agent demonstration"""
    print("🎯 Frontend Specialist Agent Comprehensive Demo")
    print("Issue #72 Implementation Demonstration")
    print("=" * 60)
    
    try:
        # Run all demos
        demo_react_analysis()
        demo_suggestions()
        demo_accessibility_focus()
        demo_performance_analysis()
        
        print("\n" + "=" * 60)
        print("✅ Frontend Specialist Agent Demo Complete!")
        print("\n🎉 Key Features Demonstrated:")
        print("  • React & Vue component analysis")
        print("  • WCAG 2.1 AA accessibility validation")
        print("  • Performance optimization suggestions")
        print("  • Multi-dimensional suggestion prioritization")
        print("  • Framework-specific pattern detection")
        print("  • Comprehensive error analysis")
        
        print("\n📚 Usage in RIF System:")
        print("  1. Create agent: agent = FrontendSpecialistAgent()")
        print("  2. Analyze code: analysis = agent.analyze_component(code, context)")
        print("  3. Get suggestions: suggestions = agent.suggest_improvements(code)")
        print("  4. Apply recommendations based on priority and effort")
        
        print("\n🔗 Integration Points:")
        print("  • Ready for DomainAgentFactory integration (Issue #71)")
        print("  • Knowledge base pattern loading and learning storage")
        print("  • Analysis history tracking and performance metrics")
        print("  • RIF workflow checkpoint management")
        
    except Exception as e:
        print(f"\n❌ Demo failed with error: {e}")
        import traceback
        traceback.print_exc()
        return 1
    
    return 0

if __name__ == '__main__':
    exit(main())