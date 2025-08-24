#!/usr/bin/env python3
"""
Test Backend Specialist Agent - Issue #73
Comprehensive tests for backend specialist agent functionality.
"""

import unittest
import sys
import json
from pathlib import Path
from unittest.mock import Mock, patch

# Add the commands directory to the path
sys.path.append(str(Path(__file__).parent.parent / "claude" / "commands"))
from backend_specialist_agent import BackendSpecialistAgent

class TestBackendSpecialistAgent(unittest.TestCase):
    """Test cases for Backend Specialist Agent"""
    
    def setUp(self):
        """Set up test fixtures"""
        self.agent = BackendSpecialistAgent()
        
        # Sample Flask API code
        self.sample_flask_code = '''
from flask import Flask, request, jsonify
import sqlite3

app = Flask(__name__)

@app.route('/api/users', methods=['GET'])
def get_users():
    conn = sqlite3.connect('users.db')
    cursor = conn.cursor()
    cursor.execute("SELECT * FROM users")
    users = cursor.fetchall()
    conn.close()
    return jsonify(users)

@app.route('/api/users/<int:user_id>', methods=['GET'])
def get_user(user_id):
    conn = sqlite3.connect('users.db')
    cursor = conn.cursor()
    cursor.execute("SELECT * FROM users WHERE id = ?", (user_id,))
    user = cursor.fetchone()
    conn.close()
    if user:
        return jsonify(user)
    return jsonify({'error': 'User not found'}), 404
        '''
        
        # Sample problematic code for testing
        self.problematic_code = '''
@app.route('/api/posts')
def get_posts():
    posts = []
    for post_id in get_all_post_ids():
        post = db.query("SELECT * FROM posts WHERE id = " + str(post_id))
        posts.append(post)
    return posts
        '''
        
        # Sample database schema
        self.sample_schema = '''
CREATE TABLE users (
    id INTEGER PRIMARY KEY,
    username VARCHAR(50) NOT NULL,
    email VARCHAR(100),
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

CREATE TABLE posts (
    id INTEGER PRIMARY KEY,
    user_id INTEGER,
    title VARCHAR(200) NOT NULL,
    content TEXT,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    FOREIGN KEY (user_id) REFERENCES users(id)
);
        '''
    
    def test_agent_initialization(self):
        """Test that the agent initializes correctly"""
        self.assertEqual(self.agent.domain, 'backend')
        self.assertIn('api_development', self.agent.capabilities)
        self.assertIn('database_design', self.agent.capabilities)
        self.assertIn('caching_strategies', self.agent.capabilities)
        self.assertIn('scaling_patterns', self.agent.capabilities)
        
    def test_component_analysis(self):
        """Test comprehensive component analysis"""
        result = self.agent.analyze_component(self.sample_flask_code)
        
        # Check result structure
        self.assertIn('component_info', result)
        self.assertIn('issues', result)
        self.assertIn('metrics', result)
        self.assertIn('confidence', result)
        self.assertIn('recommendations', result)
        
        # Check component identification
        component_info = result['component_info']
        self.assertEqual(component_info['framework'], 'flask')
        self.assertEqual(component_info['language'], 'python')
        self.assertTrue(component_info['has_api'])
        self.assertTrue(component_info['has_database'])
        
        # Check metrics
        metrics = result['metrics']
        self.assertIn('lines_of_code', metrics)
        self.assertIn('api_endpoints', metrics)
        self.assertIn('database_queries', metrics)
        self.assertGreater(metrics['api_endpoints'], 0)
        
    def test_api_analysis(self):
        """Test API analysis functionality"""
        result = self.agent.analyze_api(self.sample_flask_code)
        
        self.assertIn('analysis_results', result)
        self.assertIn('api_score', result)
        self.assertIn('recommendations', result)
        
        analysis_results = result['analysis_results']
        self.assertIn('rest_compliance', analysis_results)
        self.assertIn('performance', analysis_results)
        self.assertIn('security', analysis_results)
        self.assertIn('scalability', analysis_results)
        
    def test_rest_compliance_check(self):
        """Test REST compliance checking"""
        result = self.agent.check_rest_compliance(self.sample_flask_code)
        
        self.assertIn('compliance_score', result)
        self.assertIn('issues', result)
        self.assertIn('recommendations', result)
        
        # Score should be between 0 and 100
        self.assertGreaterEqual(result['compliance_score'], 0)
        self.assertLessEqual(result['compliance_score'], 100)
        
    def test_performance_analysis(self):
        """Test performance analysis"""
        result = self.agent.analyze_performance(self.problematic_code)
        
        self.assertIn('performance_score', result)
        self.assertIn('issues', result)
        
        # Should detect N+1 query problem
        issues = result['issues']
        n_plus_one_issues = [issue for issue in issues if 'N+1' in issue.get('message', '')]
        self.assertGreater(len(n_plus_one_issues), 0)
        
    def test_security_check(self):
        """Test API security checking"""
        result = self.agent.check_api_security(self.problematic_code)
        
        self.assertIn('security_score', result)
        self.assertIn('issues', result)
        
        # Should detect SQL injection vulnerability
        issues = result['issues']
        sql_injection_issues = [issue for issue in issues if 'SQL injection' in issue.get('message', '')]
        self.assertGreater(len(sql_injection_issues), 0)
        
    def test_database_optimization(self):
        """Test database optimization recommendations"""
        # Mock schema and queries
        schema_dict = {'tables': {'users': {'columns': []}}}
        queries_dict = {'parsed_queries': []}
        
        result = self.agent.optimize_database(schema_dict, context={'database_type': 'postgresql'})
        
        self.assertIn('optimizations', result)
        self.assertIn('schema_analysis', result)
        self.assertIn('query_analysis', result)
        self.assertIn('estimated_improvement', result)
        
    def test_index_recommendations(self):
        """Test index recommendation logic"""
        schema = {'tables': {'users': {'columns': []}}}
        queries = {
            'parsed_queries': [
                {
                    'table': 'users',
                    'where_columns': ['email', 'status'],
                    'join_columns': []
                }
            ]
        }
        
        recommendations = self.agent.recommend_indexes(schema, queries)
        
        # Should recommend indexes for WHERE columns
        self.assertGreater(len(recommendations), 0)
        
        email_index = None
        for rec in recommendations:
            if 'email' in rec.get('columns', []):
                email_index = rec
                break
        
        self.assertIsNotNone(email_index)
        self.assertEqual(email_index['type'], 'index_recommendation')
        self.assertIn('sql', email_index)
        
    def test_caching_strategy_suggestions(self):
        """Test caching strategy suggestions"""
        cache_code = '''
@app.route('/api/popular-posts')
def get_popular_posts():
    # This is called frequently and data doesn't change often
    posts = db.query("SELECT * FROM posts ORDER BY views DESC LIMIT 10")
    return jsonify(posts)
        '''
        
        result = self.agent.suggest_caching_strategy(cache_code)
        
        self.assertIn('current_caching', result)
        self.assertIn('recommended_strategies', result)
        self.assertIn('cacheable_operations', result)
        
    def test_scaling_assessment(self):
        """Test scaling potential assessment"""
        scaling_code = '''
from flask import Flask, session
import sqlite3

app = Flask(__name__)

@app.route('/api/user-data')
def get_user_data():
    user_id = session['user_id']  # Stateful operation
    conn = sqlite3.connect('app.db')  # No connection pooling
    # ... rest of the code
        '''
        
        result = self.agent.assess_scaling_potential(scaling_code)
        
        self.assertIn('architecture_analysis', result)
        self.assertIn('bottlenecks', result)
        self.assertIn('horizontal_scaling', result)
        self.assertIn('vertical_scaling', result)
        self.assertIn('scaling_strategy', result)
        
    def test_scalability_check(self):
        """Test scalability assessment"""
        stateful_code = '''
session['user_data'] = user_info
global_cache = {}
        '''
        
        result = self.agent.assess_scalability(stateful_code)
        
        self.assertIn('scalability_score', result)
        self.assertIn('issues', result)
        
        # Should detect stateful operations
        issues = result['issues']
        stateful_issues = [issue for issue in issues if 'stateful' in issue.get('message', '').lower()]
        self.assertGreater(len(stateful_issues), 0)
        
    def test_improvement_suggestions(self):
        """Test improvement suggestions generation"""
        result = self.agent.suggest_improvements(self.problematic_code)
        
        self.assertIn('suggestions', result)
        self.assertIn('prioritized', result)
        self.assertIn('implementation_roadmap', result)
        self.assertIn('estimated_impact', result)
        
        suggestions = result['suggestions']
        self.assertIn('api_improvements', suggestions)
        self.assertIn('database_improvements', suggestions)
        self.assertIn('security_improvements', suggestions)
        
    def test_backend_type_identification(self):
        """Test backend technology identification"""
        # Test Flask identification
        flask_info = self.agent._identify_backend_type(self.sample_flask_code, {})
        self.assertEqual(flask_info['framework'], 'flask')
        self.assertEqual(flask_info['language'], 'python')
        self.assertTrue(flask_info['has_api'])
        
        # Test Express.js identification
        express_code = '''
const express = require('express');
const app = express();
        '''
        express_info = self.agent._identify_backend_type(express_code, {})
        self.assertEqual(express_info['framework'], 'express')
        self.assertEqual(express_info['language'], 'javascript')
        
    def test_metrics_calculation(self):
        """Test backend metrics calculation"""
        metrics = self.agent._calculate_backend_metrics(self.sample_flask_code, {})
        
        self.assertIn('lines_of_code', metrics)
        self.assertIn('cyclomatic_complexity', metrics)
        self.assertIn('api_endpoints', metrics)
        self.assertIn('database_queries', metrics)
        self.assertIn('external_dependencies', metrics)
        
        # Check reasonable values
        self.assertGreater(metrics['lines_of_code'], 0)
        self.assertGreater(metrics['api_endpoints'], 0)
        self.assertGreater(metrics['database_queries'], 0)
        
    def test_confidence_score_calculation(self):
        """Test confidence score calculation"""
        issues = [{'severity': 'high'}, {'severity': 'medium'}]
        metrics = {'lines_of_code': 100, 'cyclomatic_complexity': 5}
        
        confidence = self.agent._calculate_confidence_score(issues, metrics)
        
        self.assertIsInstance(confidence, float)
        self.assertGreaterEqual(confidence, 0.0)
        self.assertLessEqual(confidence, 1.0)
        
    def test_pattern_loading(self):
        """Test that patterns are loaded correctly"""
        api_patterns = self.agent._load_api_patterns()
        db_patterns = self.agent._load_database_patterns()
        cache_patterns = self.agent._load_caching_patterns()
        scaling_patterns = self.agent._load_scaling_patterns()
        
        self.assertIn('rest_principles', api_patterns)
        self.assertIn('indexing_strategies', db_patterns)
        self.assertIn('cache_types', cache_patterns)
        self.assertIn('horizontal_scaling', scaling_patterns)
        
    def test_domain_info(self):
        """Test getting domain information"""
        info = self.agent.get_domain_info()
        
        self.assertEqual(info['domain'], 'backend')
        self.assertIn('api_development', info['capabilities'])
        self.assertIn('agent_id', info)
        self.assertIn('created_at', info)
        
    def test_capability_validation(self):
        """Test capability validation"""
        self.assertTrue(self.agent.validate_capability('api_development'))
        self.assertTrue(self.agent.validate_capability('database_design'))
        self.assertFalse(self.agent.validate_capability('nonexistent_capability'))
        
    def test_analysis_recording(self):
        """Test analysis recording functionality"""
        initial_count = len(self.agent.analysis_history)
        
        # Perform an analysis
        self.agent.analyze_component(self.sample_flask_code)
        
        # Check that analysis was recorded
        self.assertGreater(len(self.agent.analysis_history), initial_count)
        
        # Check the recorded analysis structure
        latest_analysis = self.agent.analysis_history[-1]
        self.assertIn('timestamp', latest_analysis)
        self.assertIn('analysis_type', latest_analysis)
        self.assertIn('results_summary', latest_analysis)
        
    def test_performance_metrics(self):
        """Test performance metrics generation"""
        # Perform some analyses first
        self.agent.analyze_component(self.sample_flask_code)
        self.agent.analyze_api(self.sample_flask_code)
        
        metrics = self.agent.get_performance_metrics()
        
        self.assertIn('total_analyses', metrics)
        self.assertIn('recent_analyses', metrics)
        self.assertGreater(metrics['total_analyses'], 0)
        
    def test_edge_cases(self):
        """Test edge cases and error handling"""
        # Test with empty code
        result = self.agent.analyze_component("")
        self.assertIn('issues', result)
        self.assertIn('metrics', result)
        
        # Test with invalid input
        result = self.agent.analyze_api(None)
        self.assertIsInstance(result, dict)
        
    def test_specific_backend_scenarios(self):
        """Test specific backend development scenarios"""
        
        # Scenario 1: Missing database connection pooling
        no_pooling_code = '''
import sqlite3
def get_data():
    conn = sqlite3.connect('db.sqlite')
    # Direct connection, no pooling
        '''
        
        result = self.agent.assess_scalability(no_pooling_code)
        scalability_issues = [issue for issue in result['issues'] 
                            if 'connection' in issue.get('message', '').lower()]
        self.assertGreater(len(scalability_issues), 0)
        
        # Scenario 2: Missing rate limiting
        no_rate_limit_code = '''
@app.route('/api/data')
def get_data():
    return expensive_operation()
        '''
        
        security_result = self.agent.check_api_security(no_rate_limit_code)
        rate_limit_issues = [issue for issue in security_result['issues'] 
                           if 'rate limit' in issue.get('message', '').lower()]
        self.assertGreater(len(rate_limit_issues), 0)
        
    def test_integration_with_factory_pattern(self):
        """Test integration patterns with the domain agent factory"""
        # Verify agent can be instantiated correctly
        self.assertIsInstance(self.agent.domain, str)
        self.assertIsInstance(self.agent.capabilities, list)
        self.assertIsNotNone(self.agent.agent_id)
        
        # Test that agent follows expected interface
        required_methods = ['analyze_component', 'suggest_improvements']
        for method in required_methods:
            self.assertTrue(hasattr(self.agent, method))
            self.assertTrue(callable(getattr(self.agent, method)))

class TestBackendSpecialistIntegration(unittest.TestCase):
    """Integration tests for Backend Specialist Agent"""
    
    def setUp(self):
        """Set up integration test fixtures"""
        self.agent = BackendSpecialistAgent()
    
    def test_real_world_api_analysis(self):
        """Test with a more realistic API example"""
        realistic_api = '''
from flask import Flask, request, jsonify, g
from werkzeug.security import check_password_hash
import jwt
import redis
from sqlalchemy import create_engine
from sqlalchemy.pool import QueuePool

app = Flask(__name__)
redis_client = redis.Redis(host='localhost', port=6379, db=0)
engine = create_engine('postgresql://user:pass@localhost/db', 
                      poolclass=QueuePool, pool_size=10)

@app.before_request
def authenticate():
    token = request.headers.get('Authorization')
    if token:
        try:
            payload = jwt.decode(token, app.config['SECRET_KEY'], algorithms=['HS256'])
            g.user_id = payload['user_id']
        except jwt.InvalidTokenError:
            return jsonify({'error': 'Invalid token'}), 401

@app.route('/api/v1/users', methods=['GET'])
def get_users():
    page = request.args.get('page', 1, type=int)
    per_page = request.args.get('per_page', 20, type=int)
    
    # Check cache first
    cache_key = f"users:page:{page}:per_page:{per_page}"
    cached_result = redis_client.get(cache_key)
    if cached_result:
        return cached_result
    
    # Query with pagination
    offset = (page - 1) * per_page
    with engine.connect() as conn:
        result = conn.execute(
            "SELECT id, username, email FROM users LIMIT %s OFFSET %s",
            (per_page, offset)
        )
        users = [dict(row) for row in result]
    
    # Cache the result
    redis_client.setex(cache_key, 300, jsonify(users).get_data())
    return jsonify(users)
        '''
        
        result = self.agent.analyze_component(realistic_api)
        
        # Should have fewer issues due to better practices
        high_severity_issues = [issue for issue in result['issues'] 
                               if issue.get('severity') == 'high']
        
        # Should detect good practices
        component_info = result['component_info']
        self.assertTrue(component_info['has_database'])
        self.assertTrue(component_info['has_caching'])
        
        # API analysis should show good practices
        api_analysis = self.agent.analyze_api(realistic_api)
        self.assertGreaterEqual(api_analysis['api_score'], 75)  # Should have good score
    
    def test_microservices_analysis(self):
        """Test analysis of microservices patterns"""
        microservice_code = '''
from flask import Flask
import requests
from circuitbreaker import circuit

app = Flask(__name__)

@circuit(failure_threshold=5, recovery_timeout=30)
def call_user_service(user_id):
    response = requests.get(f'http://user-service/api/users/{user_id}')
    return response.json()

@circuit(failure_threshold=3, recovery_timeout=20)
def call_order_service(user_id):
    response = requests.get(f'http://order-service/api/orders?user_id={user_id}')
    return response.json()

@app.route('/api/user-summary/<int:user_id>')
def get_user_summary(user_id):
    try:
        user_data = call_user_service(user_id)
        orders = call_order_service(user_id)
        return jsonify({
            'user': user_data,
            'orders': orders,
            'summary': generate_summary(user_data, orders)
        })
    except Exception as e:
        return jsonify({'error': 'Service temporarily unavailable'}), 503
        '''
        
        result = self.agent.analyze_component(microservice_code)
        
        # Should identify microservices patterns
        self.assertIn('microservices', result['component_info'].get('architecture_pattern', '').lower() 
                     or str(result).lower())

if __name__ == '__main__':
    unittest.main()