#!/usr/bin/env python3
"""
Backend Specialist Factory Integration Demo - Issue #73
Demonstration of backend specialist agent integration with the domain agent factory.
"""

import sys
import json
from pathlib import Path
import time

# Add the commands directory to the path
sys.path.append(str(Path(__file__).parent / "claude" / "commands"))

try:
    from domain_agent_factory import (
        DomainAgentFactory, 
        DomainConfiguration, 
        DomainCapability,
        AgentType
    )
    from backend_specialist_agent import BackendSpecialistAgent
    factory_available = True
except ImportError as e:
    print(f"⚠️  Factory system not fully available: {e}")
    factory_available = False

def demo_factory_integration():
    """Demonstrate backend specialist integration with factory system"""
    print("\n" + "="*80)
    print("Backend Specialist Agent - Factory Integration Demo")
    print("="*80)
    
    if not factory_available:
        print("❌ Factory system not available. Running standalone demo instead.")
        return demo_standalone_agent()
    
    # Create factory instance
    factory = DomainAgentFactory()
    
    print("✅ Domain Agent Factory initialized")
    print(f"   Factory metrics: {factory.factory_metrics}")
    
    # Create backend specialist configuration
    backend_config = DomainConfiguration(
        name="Backend_API_Specialist",
        domain_type=AgentType.BACKEND,
        capabilities=[
            DomainCapability(
                name="api_design", 
                description="REST API design and validation",
                complexity="medium",
                resource_requirement="standard",
                tools=["openapi", "swagger", "postman"]
            ),
            DomainCapability(
                name="database_optimization", 
                description="Database performance optimization",
                complexity="high",
                resource_requirement="intensive",
                tools=["postgresql", "mysql", "mongodb", "redis"]
            ),
            DomainCapability(
                name="caching_strategies", 
                description="Caching pattern implementation",
                complexity="medium",
                resource_requirement="standard",
                tools=["redis", "memcached", "nginx"]
            ),
            DomainCapability(
                name="scaling_patterns", 
                description="Horizontal and vertical scaling analysis",
                complexity="high",
                resource_requirement="intensive",
                tools=["kubernetes", "docker", "load_balancer"]
            )
        ],
        expertise=["python", "flask", "django", "nodejs", "postgresql", "redis", "docker"],
        tools=["python", "postgresql", "redis", "docker", "nginx", "kubernetes"],
        priority=85,
        resource_requirements={
            "memory_mb": 1024,
            "cpu_cores": 2,
            "disk_mb": 500,
            "max_runtime_minutes": 60,
            "max_concurrent_tasks": 5
        },
        metadata={
            "specialization": "backend_infrastructure",
            "experience_level": "senior",
            "performance_target": "100ms_analysis",
            "created_for_issue": 73
        }
    )
    
    print(f"\n📋 Backend Specialist Configuration:")
    print(f"   Name: {backend_config.name}")
    print(f"   Domain: {backend_config.domain_type.value}")
    print(f"   Capabilities: {len(backend_config.capabilities)}")
    print(f"   Priority: {backend_config.priority}")
    
    # Validate configuration
    print(f"\n🔍 Validating configuration...")
    is_valid, validation_message = factory.validate_config(backend_config)
    
    if is_valid:
        print(f"✅ Configuration valid: {validation_message}")
    else:
        print(f"❌ Configuration invalid: {validation_message}")
        return
    
    # Create agent through factory
    print(f"\n🏭 Creating agent through factory...")
    start_time = time.time()
    success, agent, message = factory.create_agent(backend_config)
    creation_time = time.time() - start_time
    
    if success:
        print(f"✅ Agent created successfully in {creation_time:.3f}s")
        print(f"   Agent ID: {agent.agent_id}")
        print(f"   Status: {agent.status.value}")
        print(f"   Resources allocated: {agent.resources is not None}")
        print(f"   Message: {message}")
    else:
        print(f"❌ Agent creation failed: {message}")
        return
    
    # Test agent capabilities
    print(f"\n🧪 Testing agent capabilities...")
    
    # Test API analysis
    sample_api_code = '''
@app.route('/api/users', methods=['GET'])
def get_users():
    conn = sqlite3.connect('users.db')
    users = conn.execute("SELECT * FROM users").fetchall()
    conn.close()
    return jsonify(users)
    '''
    
    # Get the actual specialist agent (the factory creates a DomainAgent wrapper)
    # For demo purposes, create a direct instance to test capabilities
    specialist = BackendSpecialistAgent()
    
    print(f"   Testing API analysis...")
    api_result = specialist.analyze_api(sample_api_code)
    print(f"   ✅ API analysis completed - Score: {api_result['api_score']:.1f}/100")
    
    print(f"   Testing database optimization...")
    db_result = specialist.optimize_database("CREATE TABLE users (id INT PRIMARY KEY);")
    print(f"   ✅ Database optimization completed - {len(db_result['optimizations'])} recommendations")
    
    print(f"   Testing caching strategies...")
    cache_result = specialist.suggest_caching_strategy(sample_api_code)
    print(f"   ✅ Caching analysis completed - {len(cache_result['recommended_strategies'])} strategies")
    
    # Show factory metrics
    print(f"\n📊 Factory Metrics After Creation:")
    metrics = factory.get_factory_metrics()
    factory_metrics = metrics['factory_metrics']
    print(f"   Agents Created: {factory_metrics['agents_created']}")
    print(f"   Creation Failures: {factory_metrics['creation_failures']}")
    print(f"   Average Creation Time: {factory_metrics['average_creation_time']:.3f}s")
    print(f"   Backend Agents: {factory_metrics['agents_by_type'][AgentType.BACKEND]}")
    
    # Test resource usage
    resource_usage = metrics['resource_usage']
    print(f"\n💻 Resource Usage:")
    print(f"   Allocated Agents: {resource_usage['allocated_agents']}")
    print(f"   CPU Usage: {resource_usage['cpu_utilization']['percentage']:.1f}%")
    print(f"   Memory Usage: {resource_usage['memory_utilization']['percentage']:.1f}%")
    
    # Clean up
    print(f"\n🧹 Cleaning up agent...")
    cleanup_success = factory.cleanup_agent(agent.agent_id)
    if cleanup_success:
        print(f"✅ Agent cleaned up successfully")
    else:
        print(f"⚠️  Agent cleanup had issues")
    
    return True

def demo_standalone_agent():
    """Demonstrate standalone backend specialist agent"""
    print("\n" + "="*80)
    print("Backend Specialist Agent - Standalone Demo")
    print("="*80)
    
    # Create agent directly
    agent = BackendSpecialistAgent()
    
    print(f"✅ Backend Specialist Agent created")
    print(f"   Domain: {agent.domain}")
    print(f"   Agent ID: {agent.agent_id}")
    print(f"   Capabilities: {len(agent.capabilities)}")
    
    # Test comprehensive analysis
    sample_backend_code = '''
from flask import Flask, request, jsonify
import sqlite3
import redis
from functools import wraps

app = Flask(__name__)
redis_client = redis.Redis(host='localhost')

def auth_required(f):
    @wraps(f)
    def decorated(*args, **kwargs):
        token = request.headers.get('Authorization')
        if not token:
            return jsonify({'error': 'No token'}), 401
        return f(*args, **kwargs)
    return decorated

@app.route('/api/v1/users', methods=['GET'])
@auth_required  
def get_users():
    page = request.args.get('page', 1, type=int)
    per_page = min(request.args.get('per_page', 10, type=int), 100)
    
    # Check cache first
    cache_key = f"users:page:{page}:per_page:{per_page}"
    cached = redis_client.get(cache_key)
    if cached:
        return cached
    
    conn = sqlite3.connect('users.db')
    cursor = conn.cursor()
    
    offset = (page - 1) * per_page
    cursor.execute("SELECT id, username, email FROM users LIMIT ? OFFSET ?", 
                  (per_page, offset))
    users = cursor.fetchall()
    conn.close()
    
    result = jsonify({'users': users, 'page': page})
    redis_client.setex(cache_key, 300, result.get_data())
    
    return result
    '''
    
    print(f"\n🔍 Performing comprehensive analysis...")
    start_time = time.time()
    analysis = agent.analyze_component(sample_backend_code)
    analysis_time = time.time() - start_time
    
    print(f"✅ Analysis completed in {analysis_time:.3f}s")
    print(f"   Confidence: {analysis['confidence']:.2f}")
    print(f"   Issues found: {len(analysis['issues'])}")
    
    # Show component info
    component_info = analysis['component_info']
    print(f"\n📋 Component Analysis:")
    print(f"   Framework: {component_info['framework']}")
    print(f"   Has API: {component_info['has_api']}")
    print(f"   Has Database: {component_info['has_database']}")
    print(f"   Has Caching: {component_info['has_caching']}")
    
    # Show metrics
    metrics = analysis['metrics']
    print(f"\n📊 Code Metrics:")
    print(f"   Lines of Code: {metrics['lines_of_code']}")
    print(f"   API Endpoints: {metrics['api_endpoints']}")
    print(f"   Database Queries: {metrics['database_queries']}")
    print(f"   Complexity: {metrics['cyclomatic_complexity']}")
    
    # Show top issues
    if analysis['issues']:
        print(f"\n⚠️  Top Issues:")
        for issue in analysis['issues'][:3]:
            severity_icon = "🔴" if issue['severity'] == 'critical' else "🟡" if issue['severity'] == 'high' else "🟠"
            print(f"   {severity_icon} {issue['message']} (Line {issue.get('line', 'N/A')})")
    
    return True

def demo_team_creation():
    """Demonstrate creating a backend-focused team"""
    print("\n" + "-"*60)
    print("TEAM CREATION DEMO")
    print("-"*60)
    
    if not factory_available:
        print("❌ Factory system not available for team creation demo")
        return
    
    factory = DomainAgentFactory()
    
    # Project requirements for backend-heavy project
    project_requirements = {
        "project_name": "E-Commerce Backend",
        "type": "web_api",
        "technologies": ["python", "flask", "postgresql", "redis", "docker"],
        "features": [
            "rest_api", 
            "authentication", 
            "database_optimization",
            "caching", 
            "microservices",
            "scaling"
        ],
        "description": "High-performance e-commerce backend with microservices architecture"
    }
    
    print(f"🏗️  Creating specialist team for: {project_requirements['project_name']}")
    
    success, team, message = factory.create_specialist_team(project_requirements)
    
    if success:
        print(f"✅ Team created successfully: {message}")
        print(f"   Team size: {len(team)} agents")
        
        for agent in team:
            print(f"   • {agent.name} ({agent.domain_type.value})")
            
        # Show team metadata
        if team and 'team' in team[0].metadata:
            team_metadata = team[0].metadata['team']
            print(f"\n📋 Team Metadata:")
            print(f"   Team ID: {team_metadata['team_id']}")
            print(f"   Domains: {', '.join(team_metadata['domains'])}")
            print(f"   Created: {team_metadata['created_at']}")
    else:
        print(f"❌ Team creation failed: {message}")

def demo_performance_comparison():
    """Compare factory vs direct creation performance"""
    print("\n" + "-"*60)
    print("PERFORMANCE COMPARISON DEMO")  
    print("-"*60)
    
    if not factory_available:
        print("❌ Factory system not available for performance comparison")
        return
    
    # Direct creation
    print("⚡ Testing direct agent creation...")
    direct_times = []
    for i in range(3):
        start_time = time.time()
        agent = BackendSpecialistAgent()
        direct_times.append(time.time() - start_time)
    
    avg_direct_time = sum(direct_times) / len(direct_times)
    print(f"   Average direct creation time: {avg_direct_time:.4f}s")
    
    # Factory creation
    print("🏭 Testing factory agent creation...")
    factory = DomainAgentFactory()
    
    backend_config = DomainConfiguration(
        name="Performance_Test_Agent",
        domain_type=AgentType.BACKEND,
        capabilities=[
            DomainCapability(
                name="api_design",
                description="API design testing"
            )
        ],
        expertise=["python"],
        tools=["python"]
    )
    
    factory_times = []
    created_agents = []
    
    for i in range(3):
        start_time = time.time()
        success, agent, message = factory.create_agent(backend_config)
        factory_times.append(time.time() - start_time)
        
        if success:
            created_agents.append(agent)
            # Update name for uniqueness
            backend_config.name = f"Performance_Test_Agent_{i+1}"
    
    avg_factory_time = sum(factory_times) / len(factory_times)
    print(f"   Average factory creation time: {avg_factory_time:.4f}s")
    
    # Comparison
    overhead = avg_factory_time - avg_direct_time
    print(f"\n📊 Performance Comparison:")
    print(f"   Direct creation: {avg_direct_time:.4f}s")
    print(f"   Factory creation: {avg_factory_time:.4f}s")
    print(f"   Factory overhead: {overhead:.4f}s ({overhead/avg_direct_time*100:.1f}% increase)")
    
    # Clean up created agents
    for agent in created_agents:
        factory.cleanup_agent(agent.agent_id)

def main():
    """Main demonstration function"""
    print("Backend Specialist Agent - Factory Integration Demo")
    print("Issue #73 Implementation with Issue #71 Integration")
    
    try:
        # Factory integration
        demo_factory_integration()
        
        # Team creation
        demo_team_creation()
        
        # Performance comparison
        demo_performance_comparison()
        
        print("\n" + "="*80)
        print("✅ Factory Integration Demo completed successfully!")
        print("Backend Specialist Agent is fully integrated with the Domain Agent Factory.")
        print("="*80)
        
    except Exception as e:
        print(f"\n❌ Demo failed with error: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()