#!/usr/bin/env python3
"""
Backend Specialist Agent Demo - Issue #73
Demonstration of backend specialist agent capabilities with real-world examples.
"""

import sys
from pathlib import Path
import json
import time

# Add the commands directory to the path
sys.path.append(str(Path(__file__).parent / "claude" / "commands"))
from backend_specialist_agent import BackendSpecialistAgent

def demo_basic_functionality():
    """Demonstrate basic backend specialist functionality"""
    print("\n" + "="*80)
    print("Backend Specialist Agent - Basic Functionality Demo")
    print("="*80)
    
    # Initialize the agent
    agent = BackendSpecialistAgent()
    
    print(f"Agent initialized:")
    print(f"  Domain: {agent.domain}")
    print(f"  Capabilities: {', '.join(agent.capabilities)}")
    print(f"  Agent ID: {agent.agent_id}")
    
    return agent

def demo_api_analysis():
    """Demonstrate API analysis capabilities"""
    print("\n" + "-"*60)
    print("API ANALYSIS DEMO")
    print("-"*60)
    
    agent = BackendSpecialistAgent()
    
    # Sample API with various issues
    problematic_api = '''
from flask import Flask, request
import sqlite3

app = Flask(__name__)

@app.route('/getUserData')  # Non-RESTful URL
def getUserData():
    user_id = request.args.get('id')
    # SQL injection vulnerability
    query = "SELECT * FROM users WHERE id = " + user_id
    conn = sqlite3.connect('users.db')
    result = conn.execute(query).fetchall()
    conn.close()
    return result  # No proper JSON response

@app.route('/api/posts')
def getAllPosts():
    conn = sqlite3.connect('posts.db') 
    # No pagination - will return all records
    posts = conn.execute("SELECT * FROM posts").fetchall()
    conn.close()
    return posts
    '''
    
    print("Analyzing problematic API code...")
    start_time = time.time()
    result = agent.analyze_api(problematic_api)
    analysis_time = time.time() - start_time
    
    print(f"Analysis completed in {analysis_time:.3f} seconds")
    print(f"API Score: {result['api_score']:.1f}/100")
    
    print("\nIssues Found:")
    for issue in result['issues']:
        severity_icon = "🔴" if issue['severity'] == 'critical' else "🟡" if issue['severity'] == 'high' else "🟠"
        print(f"  {severity_icon} {issue['message']} (Line {issue.get('line', 'N/A')})")
    
    print("\nREST Compliance Analysis:")
    rest_analysis = result['analysis_results']['rest_compliance']
    print(f"  Compliance Score: {rest_analysis['compliance_score']}/100")
    
    print("\nSecurity Analysis:")
    security_analysis = result['analysis_results']['security']
    print(f"  Security Score: {security_analysis['security_score']}/100")
    
    return result

def demo_database_optimization():
    """Demonstrate database optimization capabilities"""
    print("\n" + "-"*60)
    print("DATABASE OPTIMIZATION DEMO")
    print("-"*60)
    
    agent = BackendSpecialistAgent()
    
    # Sample schema and queries
    sample_schema = '''
    CREATE TABLE users (
        id INTEGER PRIMARY KEY,
        username VARCHAR(50) NOT NULL,
        email VARCHAR(100),
        first_name VARCHAR(50),
        last_name VARCHAR(50),
        created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
    );
    
    CREATE TABLE posts (
        id INTEGER PRIMARY KEY,
        user_id INTEGER,
        title VARCHAR(200) NOT NULL,
        content TEXT,
        published_at TIMESTAMP,
        view_count INTEGER DEFAULT 0,
        FOREIGN KEY (user_id) REFERENCES users(id)
    );
    '''
    
    sample_queries = [
        "SELECT * FROM users WHERE email = 'user@example.com'",
        "SELECT p.*, u.username FROM posts p JOIN users u ON p.user_id = u.id WHERE p.published_at > '2023-01-01'",
        "SELECT COUNT(*) FROM posts WHERE user_id = 123",
        "SELECT * FROM posts ORDER BY view_count DESC LIMIT 10"
    ]
    
    print("Analyzing database schema and queries...")
    result = agent.optimize_database(sample_schema, sample_queries)
    
    print(f"\nFound {len(result['optimizations'])} optimization opportunities:")
    
    for i, optimization in enumerate(result['optimizations'][:5], 1):  # Show top 5
        print(f"\n{i}. {optimization['type'].title()}")
        print(f"   Priority: {optimization['priority'].title()}")
        if 'table' in optimization:
            print(f"   Table: {optimization['table']}")
        if 'reason' in optimization:
            print(f"   Reason: {optimization['reason']}")
        if 'sql' in optimization:
            print(f"   SQL: {optimization['sql']}")
        if 'estimated_improvement' in optimization:
            print(f"   Estimated Improvement: {optimization['estimated_improvement']}")

def demo_caching_strategies():
    """Demonstrate caching strategy recommendations"""
    print("\n" + "-"*60)
    print("CACHING STRATEGY DEMO")
    print("-"*60)
    
    agent = BackendSpecialistAgent()
    
    # Sample code with caching opportunities
    cacheable_code = '''
@app.route('/api/popular-posts')
def get_popular_posts():
    # This endpoint is called frequently but data doesn't change often
    posts = db.query("""
        SELECT p.id, p.title, p.view_count, u.username 
        FROM posts p 
        JOIN users u ON p.user_id = u.id 
        ORDER BY p.view_count DESC 
        LIMIT 20
    """)
    return jsonify(posts)

@app.route('/api/user-stats/<int:user_id>')
def get_user_stats(user_id):
    # Expensive computation that could be cached
    posts_count = db.query("SELECT COUNT(*) FROM posts WHERE user_id = ?", [user_id])
    avg_views = db.query("SELECT AVG(view_count) FROM posts WHERE user_id = ?", [user_id])
    total_views = db.query("SELECT SUM(view_count) FROM posts WHERE user_id = ?", [user_id])
    
    return jsonify({
        'posts_count': posts_count,
        'avg_views': avg_views,
        'total_views': total_views
    })
    '''
    
    print("Analyzing code for caching opportunities...")
    result = agent.suggest_caching_strategy(cacheable_code)
    
    print(f"\nFound {len(result['recommended_strategies'])} caching strategies:")
    
    for strategy in result['recommended_strategies']:
        print(f"\n🔹 {strategy['type'].title()} - {strategy['pattern'].title()}")
        print(f"   Use Case: {strategy['use_case']}")
        print(f"   Benefits: {', '.join(strategy['benefits'])}")
        if 'implementation' in strategy:
            print(f"   Implementation: Available")

def demo_scaling_assessment():
    """Demonstrate scaling assessment capabilities"""
    print("\n" + "-"*60)
    print("SCALING ASSESSMENT DEMO")
    print("-"*60)
    
    agent = BackendSpecialistAgent()
    
    # Sample code with scaling challenges
    scaling_code = '''
from flask import Flask, session
import sqlite3

app = Flask(__name__)

# Global variables - not scalable
user_cache = {}
connection_pool = None

@app.route('/api/user-data')
def get_user_data():
    # Session usage - stateful, hinders horizontal scaling
    user_id = session.get('user_id')
    
    # Check global cache first
    if user_id in user_cache:
        return user_cache[user_id]
    
    # Direct database connection - no pooling
    conn = sqlite3.connect('app.db')
    cursor = conn.cursor()
    
    # Inefficient query
    cursor.execute("SELECT * FROM users WHERE id = ?", (user_id,))
    user_data = cursor.fetchone()
    
    # Additional queries in loop - N+1 problem
    cursor.execute("SELECT id FROM user_posts WHERE user_id = ?", (user_id,))
    post_ids = cursor.fetchall()
    
    posts = []
    for post_id in post_ids:
        cursor.execute("SELECT * FROM posts WHERE id = ?", (post_id[0],))
        post = cursor.fetchone()
        posts.append(post)
    
    conn.close()
    
    result = {'user': user_data, 'posts': posts}
    user_cache[user_id] = result  # Global cache update
    
    return result
    '''
    
    print("Analyzing code for scaling potential...")
    result = agent.assess_scaling_potential(scaling_code)
    
    print(f"\nScaling Assessment Results:")
    
    # Show bottlenecks
    if 'bottlenecks' in result and result['bottlenecks']:
        print(f"\nIdentified Bottlenecks:")
        for bottleneck in result['bottlenecks']:
            print(f"  ⚠️  {bottleneck}")
    
    # Show horizontal scaling assessment
    if 'horizontal_scaling' in result:
        print(f"\nHorizontal Scaling Potential: Available")
    
    # Show vertical scaling assessment  
    if 'vertical_scaling' in result:
        print(f"Vertical Scaling Potential: Available")

def demo_comprehensive_analysis():
    """Demonstrate comprehensive backend component analysis"""
    print("\n" + "-"*60)
    print("COMPREHENSIVE ANALYSIS DEMO")
    print("-"*60)
    
    agent = BackendSpecialistAgent()
    
    # Real-world Flask application example
    flask_app_code = '''
from flask import Flask, request, jsonify, g
from werkzeug.security import generate_password_hash, check_password_hash
import jwt
import redis
import psycopg2
from psycopg2 import pool
import logging
from functools import wraps
import time

app = Flask(__name__)
app.config['SECRET_KEY'] = 'your-secret-key-here'

# Connection pool for better scalability
connection_pool = pool.ThreadedConnectionPool(
    minconn=1, maxconn=20,
    host='localhost', database='myapp', user='user', password='pass'
)

# Redis for caching
redis_client = redis.Redis(host='localhost', port=6379, db=0)

def token_required(f):
    @wraps(f)
    def decorated(*args, **kwargs):
        token = request.headers.get('Authorization')
        if not token:
            return jsonify({'message': 'Token is missing'}), 401
        try:
            if token.startswith('Bearer '):
                token = token[7:]
            data = jwt.decode(token, app.config['SECRET_KEY'], algorithms=['HS256'])
            current_user_id = data['user_id']
            g.current_user_id = current_user_id
        except jwt.ExpiredSignatureError:
            return jsonify({'message': 'Token has expired'}), 401
        except jwt.InvalidTokenError:
            return jsonify({'message': 'Token is invalid'}), 401
        return f(*args, **kwargs)
    return decorated

@app.route('/api/v1/auth/login', methods=['POST'])
def login():
    try:
        data = request.get_json()
        username = data.get('username')
        password = data.get('password')
        
        # Input validation
        if not username or not password:
            return jsonify({'error': 'Username and password required'}), 400
        
        conn = connection_pool.getconn()
        cur = conn.cursor()
        
        cur.execute("SELECT id, password_hash FROM users WHERE username = %s", (username,))
        user = cur.fetchone()
        
        connection_pool.putconn(conn)
        
        if user and check_password_hash(user[1], password):
            token = jwt.encode({
                'user_id': user[0],
                'exp': time.time() + 3600  # 1 hour expiration
            }, app.config['SECRET_KEY'], algorithm='HS256')
            
            return jsonify({'token': token})
        
        return jsonify({'error': 'Invalid credentials'}), 401
        
    except Exception as e:
        logging.error(f"Login error: {e}")
        return jsonify({'error': 'Internal server error'}), 500

@app.route('/api/v1/users/<int:user_id>/posts', methods=['GET'])
@token_required
def get_user_posts(user_id):
    try:
        # Pagination parameters
        page = request.args.get('page', 1, type=int)
        per_page = min(request.args.get('per_page', 10, type=int), 100)  # Max 100
        offset = (page - 1) * per_page
        
        # Check cache first
        cache_key = f"user_posts:{user_id}:page:{page}:per_page:{per_page}"
        cached_result = redis_client.get(cache_key)
        
        if cached_result:
            return cached_result
        
        conn = connection_pool.getconn()
        cur = conn.cursor()
        
        # Optimized query with specific columns and pagination
        cur.execute("""
            SELECT p.id, p.title, p.content, p.created_at, p.view_count
            FROM posts p 
            WHERE p.user_id = %s 
            ORDER BY p.created_at DESC 
            LIMIT %s OFFSET %s
        """, (user_id, per_page, offset))
        
        posts = []
        for row in cur.fetchall():
            posts.append({
                'id': row[0],
                'title': row[1],
                'content': row[2],
                'created_at': row[3].isoformat() if row[3] else None,
                'view_count': row[4]
            })
        
        connection_pool.putconn(conn)
        
        result = jsonify({'posts': posts, 'page': page, 'per_page': per_page})
        
        # Cache for 5 minutes
        redis_client.setex(cache_key, 300, result.get_data())
        
        return result
        
    except Exception as e:
        logging.error(f"Get user posts error: {e}")
        return jsonify({'error': 'Internal server error'}), 500

if __name__ == '__main__':
    app.run(debug=False, threaded=True)
    '''
    
    print("Performing comprehensive analysis of Flask application...")
    start_time = time.time()
    result = agent.analyze_component(flask_app_code)
    analysis_time = time.time() - start_time
    
    print(f"Analysis completed in {analysis_time:.3f} seconds")
    print(f"Confidence Score: {result['confidence']:.2f}")
    
    # Component Information
    print(f"\nComponent Information:")
    component_info = result['component_info']
    print(f"  Framework: {component_info['framework']}")
    print(f"  Language: {component_info['language']}")
    print(f"  Has API: {component_info['has_api']}")
    print(f"  Has Database: {component_info['has_database']}")
    print(f"  Has Caching: {component_info['has_caching']}")
    
    # Metrics
    print(f"\nCode Metrics:")
    metrics = result['metrics']
    print(f"  Lines of Code: {metrics['lines_of_code']}")
    print(f"  API Endpoints: {metrics['api_endpoints']}")
    print(f"  Database Queries: {metrics['database_queries']}")
    print(f"  Cyclomatic Complexity: {metrics['cyclomatic_complexity']}")
    print(f"  Error Handling Blocks: {metrics['error_handling_blocks']}")
    
    # Issues Summary
    issues = result['issues']
    issue_counts = {}
    for issue in issues:
        severity = issue.get('severity', 'unknown')
        issue_counts[severity] = issue_counts.get(severity, 0) + 1
    
    print(f"\nIssues Found ({len(issues)} total):")
    for severity, count in sorted(issue_counts.items()):
        icon = "🔴" if severity == 'critical' else "🟡" if severity == 'high' else "🟠" if severity == 'medium' else "🔵"
        print(f"  {icon} {severity.title()}: {count}")
    
    # Show top issues
    if issues:
        print(f"\nTop Issues:")
        for issue in issues[:3]:
            print(f"  • {issue['message']} (Line {issue.get('line', 'N/A')})")
    
    return result

def demo_improvement_suggestions():
    """Demonstrate improvement suggestions"""
    print("\n" + "-"*60)
    print("IMPROVEMENT SUGGESTIONS DEMO")
    print("-"*60)
    
    agent = BackendSpecialistAgent()
    
    # Code that needs improvements
    improvable_code = '''
@app.route('/api/data')
def get_data():
    # No authentication
    # No input validation
    # No pagination
    # SQL injection vulnerability
    query_param = request.args.get('query')
    sql = "SELECT * FROM data WHERE name = '" + query_param + "'"
    
    conn = sqlite3.connect('app.db')  # No connection pooling
    results = conn.execute(sql).fetchall()  # No error handling
    conn.close()
    
    return results  # No JSON serialization
    '''
    
    print("Generating improvement suggestions...")
    result = agent.suggest_improvements(improvable_code)
    
    suggestions = result['suggestions']
    total_suggestions = sum(len(category_suggestions) for category_suggestions in suggestions.values())
    
    print(f"Generated {total_suggestions} improvement suggestions across {len(suggestions)} categories:")
    
    for category, category_suggestions in suggestions.items():
        if category_suggestions:
            print(f"\n{category.replace('_', ' ').title()} ({len(category_suggestions)} suggestions):")
            for i, suggestion in enumerate(category_suggestions[:2], 1):  # Show top 2 per category
                if isinstance(suggestion, dict):
                    print(f"  {i}. {suggestion.get('suggestion', 'Improvement available')}")
    
    # Show prioritized suggestions
    if 'prioritized' in result and result['prioritized']:
        print(f"\nTop Priority Suggestions:")
        for i, suggestion in enumerate(result['prioritized'][:3], 1):
            if isinstance(suggestion, dict):
                priority = suggestion.get('priority', 'medium')
                effort = suggestion.get('effort', 'unknown')
                print(f"  {i}. {suggestion.get('suggestion', 'Improvement available')}")
                print(f"     Priority: {priority.title()}, Effort: {effort.title()}")

def demo_performance_metrics():
    """Demonstrate agent performance tracking"""
    print("\n" + "-"*60)
    print("PERFORMANCE METRICS DEMO")
    print("-"*60)
    
    agent = BackendSpecialistAgent()
    
    # Perform several analyses
    sample_codes = [
        "from flask import Flask\napp = Flask(__name__)",
        "@app.route('/api/test')\ndef test(): return 'test'",
        "import sqlite3\nconn = sqlite3.connect('db.sqlite')"
    ]
    
    print("Performing multiple analyses for performance tracking...")
    for i, code in enumerate(sample_codes, 1):
        print(f"  Analysis {i}...", end="")
        agent.analyze_component(code)
        print(" Done")
    
    # Get performance metrics
    metrics = agent.get_performance_metrics()
    
    print(f"\nAgent Performance Metrics:")
    print(f"  Total Analyses: {metrics.get('total_analyses', 0)}")
    print(f"  Recent Analyses (7 days): {metrics.get('recent_analyses', 0)}")
    if 'avg_issues_per_analysis' in metrics:
        print(f"  Average Issues per Analysis: {metrics['avg_issues_per_analysis']:.1f}")
    if 'avg_suggestions_per_analysis' in metrics:
        print(f"  Average Suggestions per Analysis: {metrics['avg_suggestions_per_analysis']:.1f}")
    
    # Domain info
    print(f"\nAgent Information:")
    domain_info = agent.get_domain_info()
    print(f"  Agent ID: {domain_info['agent_id']}")
    print(f"  Domain: {domain_info['domain']}")
    print(f"  Created: {domain_info['created_at']}")
    print(f"  Analyses Performed: {domain_info['analyses_performed']}")

def main():
    """Main demonstration function"""
    print("Backend Specialist Agent - Comprehensive Demo")
    print("Issue #73 - Backend Specialist Agent Implementation")
    
    try:
        # Basic functionality
        demo_basic_functionality()
        
        # API Analysis
        demo_api_analysis()
        
        # Database Optimization  
        demo_database_optimization()
        
        # Caching Strategies
        demo_caching_strategies()
        
        # Scaling Assessment
        demo_scaling_assessment()
        
        # Comprehensive Analysis
        demo_comprehensive_analysis()
        
        # Improvement Suggestions
        demo_improvement_suggestions()
        
        # Performance Metrics
        demo_performance_metrics()
        
        print("\n" + "="*80)
        print("Demo completed successfully! ✅")
        print("Backend Specialist Agent is ready for production use.")
        print("="*80)
        
    except Exception as e:
        print(f"\n❌ Demo failed with error: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()