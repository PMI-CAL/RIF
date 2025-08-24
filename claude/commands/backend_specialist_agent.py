#!/usr/bin/env python3
"""
Backend Specialist Agent - Issue #73
Specialized agent for backend development, API design, database optimization, and scaling patterns.
"""

import json
import re
import time
from datetime import datetime, timedelta
from typing import Dict, List, Any, Optional, Set, Tuple, Union
from pathlib import Path
import logging
import hashlib

# Import the base class from the agents directory 
import sys
agents_path = str(Path(__file__).parent.parent / "agents")
if agents_path not in sys.path:
    sys.path.insert(0, agents_path)  # Insert at beginning to prioritize
# Import specifically from agents directory
import importlib.util
spec = importlib.util.spec_from_file_location("domain_agent_base", Path(__file__).parent.parent / "agents" / "domain_agent_base.py")
domain_agent_base = importlib.util.module_from_spec(spec)
spec.loader.exec_module(domain_agent_base)
DomainAgent = domain_agent_base.DomainAgent

logger = logging.getLogger(__name__)

class BackendSpecialistAgent(DomainAgent):
    """Specialized agent for backend development analysis and optimization"""
    
    def __init__(self, template_path: Optional[Union[str, Path]] = None):
        # Try to use template if provided, otherwise fallback to direct initialization
        template_file = template_path or Path("templates") / "backend-agent-template.yaml"
        
        if False and template_file.exists():
            super().__init__(template_path=template_file)
        else:
            # Fallback initialization for testing and standalone usage
            super().__init__(
                domain='backend',
                capabilities=[
                    'api_development',
                    'database_design', 
                    'caching_strategies',
                    'scaling_patterns',
                    'performance_optimization',
                    'security_audit',
                    'microservices_analysis',
                    'data_processing'
                ],
                name='backend-specialist'
            )
        
        # Backend-specific patterns and rules
        self.api_patterns = self._load_api_patterns()
        self.database_patterns = self._load_database_patterns()
        self.caching_patterns = self._load_caching_patterns()
        self.scaling_patterns = self._load_scaling_patterns()
        
        # Performance thresholds
        self.performance_thresholds = {
            'response_time_ms': 100,
            'query_time_ms': 50,
            'cache_hit_ratio': 0.85,
            'cpu_utilization': 0.75,
            'memory_utilization': 0.80
        }
        
        logger.info("Backend Specialist Agent initialized")
    

    def execute_primary_task(self, task_data: Dict[str, Any]):
        """
        Execute the primary task for the backend specialist agent
        
        Args:
            task_data: Task data containing description, context, and parameters
            
        Returns:
            TaskResult with execution outcome
        """
        from claude.agents.domain_agent_base import TaskResult, AgentStatus
        
        start_time = datetime.now()
        task_id = task_data.get('task_id', f'backend_task_{start_time.strftime("%Y%m%d_%H%M%S")}')
        task_description = task_data.get('description', '')
        context = task_data.get('context', {})
        
        result = TaskResult(
            task_id=task_id,
            status=AgentStatus.RUNNING,
            start_time=start_time
        )
        
        try:
            # Determine task type from description and execute appropriate analysis
            if 'analyze' in task_description.lower() or 'analysis' in task_description.lower():
                # Component analysis task
                component_code = task_data.get('component_code') or context.get('code', '')
                if component_code:
                    analysis_result = self.analyze_component(component_code, context)
                    result.result_data = analysis_result
                    result.confidence_score = analysis_result.get('confidence', 0.8)
                else:
                    # If no code provided, do a general backend assessment
                    result.result_data = {
                        'message': 'Backend analysis ready - provide code for detailed analysis',
                        'capabilities': self.capabilities,
                        'ready_for': ['component_analysis', 'api_analysis', 'database_optimization']
                    }
                    result.confidence_score = 0.9
                    
            elif 'api' in task_description.lower():
                # API-specific task
                api_spec = task_data.get('api_spec') or context.get('api_code', '')
                if api_spec:
                    api_result = self.analyze_api(api_spec, context)
                    result.result_data = api_result
                    result.confidence_score = api_result.get('api_score', 0.0) / 100.0
                else:
                    result.result_data = {'error': 'API specification required for API analysis'}
                    result.confidence_score = 0.0
                    
            elif 'database' in task_description.lower() or 'schema' in task_description.lower():
                # Database optimization task
                schema = task_data.get('schema') or context.get('schema')
                queries = task_data.get('queries') or context.get('queries', [])
                if schema:
                    db_result = self.optimize_database(schema, queries, context)
                    result.result_data = db_result
                    result.confidence_score = 0.85
                else:
                    result.result_data = {'error': 'Database schema required for database optimization'}
                    result.confidence_score = 0.0
                    
            elif 'cache' in task_description.lower() or 'caching' in task_description.lower():
                # Caching strategy task
                component_code = task_data.get('component_code') or context.get('code', '')
                if component_code:
                    cache_result = self.suggest_caching_strategy(component_code, context)
                    result.result_data = cache_result
                    result.confidence_score = 0.8
                else:
                    result.result_data = {'error': 'Component code required for caching analysis'}
                    result.confidence_score = 0.0
                    
            elif 'scal' in task_description.lower():  # scaling/scalability
                # Scaling assessment task
                component_code = task_data.get('component_code') or context.get('code', '')
                if component_code:
                    scaling_result = self.assess_scaling_potential(component_code, context)
                    result.result_data = scaling_result
                    result.confidence_score = 0.8
                else:
                    result.result_data = {'error': 'Component code required for scaling analysis'}
                    result.confidence_score = 0.0
                    
            elif 'improve' in task_description.lower() or 'suggestion' in task_description.lower():
                # Improvement suggestions task
                component_code = task_data.get('component_code') or context.get('code', '')
                if component_code:
                    improvement_result = self.suggest_improvements(component_code)
                    result.result_data = improvement_result
                    result.confidence_score = 0.85
                else:
                    result.result_data = {'error': 'Component code required for improvement suggestions'}
                    result.confidence_score = 0.0
            else:
                # Default comprehensive analysis
                component_code = task_data.get('component_code') or context.get('code', '')
                if component_code:
                    analysis_result = self.analyze_component(component_code, context)
                    result.result_data = analysis_result
                    result.confidence_score = analysis_result.get('confidence', 0.8)
                else:
                    result.result_data = {
                        'message': 'Backend specialist ready for task execution',
                        'supported_tasks': [
                            'component analysis', 'api analysis', 'database optimization',
                            'caching strategies', 'scaling assessment', 'improvement suggestions'
                        ],
                        'domain': self.domain,
                        'capabilities': self.capabilities
                    }
                    result.confidence_score = 0.9
            
            result.status = AgentStatus.COMPLETED
            
        except Exception as e:
            logger.error(f"Backend specialist task execution failed: {e}")
            result.status = AgentStatus.FAILED
            result.error_message = str(e)
            result.confidence_score = 0.0
        
        finally:
            result.end_time = datetime.now()
            result.duration_seconds = (result.end_time - start_time).total_seconds()
        
        return result

    def analyze_component(self, component_code: str, context: Optional[Dict] = None) -> Dict[str, Any]:
        """
        Comprehensive backend component analysis
        
        Args:
            component_code: Backend component source code
            context: Optional context (file_path, framework, database_type, etc.)
            
        Returns:
            Analysis results with issues, metrics, and recommendations
        """
        analysis_start = datetime.now()
        
        # Determine component type and framework
        component_info = self._identify_backend_type(component_code, context)
        
        issues = []
        metrics = {}
        
        # Core analysis areas
        api_issues = self.analyze_api(component_code, component_info)
        if 'issues' in api_issues:
            issues.extend(api_issues['issues'])
        
        db_issues = self.analyze_database_usage(component_code, component_info)
        if 'issues' in db_issues:
            issues.extend(db_issues['issues'])
        
        cache_issues = self.analyze_caching_strategies(component_code, component_info)
        if 'issues' in cache_issues:
            issues.extend(cache_issues['issues'])
        
        scaling_issues = self.analyze_scaling_patterns(component_code, component_info)
        if 'issues' in scaling_issues:
            issues.extend(scaling_issues['issues'])
        
        # Security analysis
        issues.extend(self._check_security_patterns(component_code, component_info))
        
        # Performance analysis
        issues.extend(self._check_performance_patterns(component_code, component_info))
        
        # Calculate metrics
        metrics = self._calculate_backend_metrics(component_code, component_info)
        
        analysis_duration = (datetime.now() - analysis_start).total_seconds()
        
        results = {
            'component_info': component_info,
            'issues': issues,
            'metrics': metrics,
            'analysis_duration': analysis_duration,
            'confidence': self._calculate_confidence_score(issues, metrics),
            'recommendations': self._generate_priority_recommendations(issues),
            'api_analysis': api_issues,
            'database_analysis': db_issues,
            'caching_analysis': cache_issues,
            'scaling_analysis': scaling_issues
        }
        
        # Record this analysis
        self.record_analysis('backend_component_analysis', results)
        
        return results
    
    def analyze_api(self, api_spec: Union[str, Dict], context: Optional[Dict] = None) -> Dict[str, Any]:
        """
        Analyze API design for REST compliance, performance, security, and scalability
        
        Args:
            api_spec: API specification (OpenAPI/Swagger or source code)
            context: Optional context information
            
        Returns:
            Comprehensive API analysis results
        """
        analysis_start = time.time()
        
        if isinstance(api_spec, str):
            # Source code analysis
            api_info = self._extract_api_info_from_code(api_spec)
        else:
            # OpenAPI/Swagger spec analysis
            api_info = api_spec
        
        analysis_results = {
            'rest_compliance': self.check_rest_compliance(api_spec),
            'performance': self.analyze_performance(api_spec, context),
            'security': self.check_api_security(api_spec),
            'scalability': self.assess_scalability(api_spec, context),
            'documentation': self._check_api_documentation(api_spec),
            'versioning': self._check_api_versioning(api_spec),
            'error_handling': self._check_error_handling(api_spec)
        }
        
        # Aggregate issues
        all_issues = []
        for category, results in analysis_results.items():
            if isinstance(results, dict) and 'issues' in results:
                for issue in results['issues']:
                    issue['category'] = f'api_{category}'
                    all_issues.append(issue)
        
        # Calculate overall API score
        api_score = self._calculate_api_score(analysis_results)
        
        return {
            'analysis_results': analysis_results,
            'issues': all_issues,
            'api_score': api_score,
            'analysis_duration': time.time() - analysis_start,
            'recommendations': self._generate_api_recommendations(analysis_results)
        }
    
    def optimize_database(self, schema: Union[str, Dict], queries: Optional[List[str]] = None, 
                         context: Optional[Dict] = None) -> Dict[str, Any]:
        """
        Optimize database schema and queries
        
        Args:
            schema: Database schema (SQL DDL or schema dict)
            queries: List of SQL queries to optimize
            context: Optional context (database_type, workload_pattern, etc.)
            
        Returns:
            Database optimization recommendations
        """
        optimizations = []
        
        # Parse schema and queries
        schema_info = self._parse_database_schema(schema)
        query_info = self._parse_queries(queries or [])
        
        # Index recommendations
        index_recommendations = self.recommend_indexes(schema_info, query_info, context)
        optimizations.extend(index_recommendations)
        
        # Query optimizations
        query_optimizations = self.optimize_queries(query_info, schema_info, context)
        optimizations.extend(query_optimizations)
        
        # Schema improvements
        schema_improvements = self.suggest_schema_improvements(schema_info, context)
        optimizations.extend(schema_improvements)
        
        # Partitioning recommendations
        partitioning_recs = self._recommend_partitioning(schema_info, context)
        optimizations.extend(partitioning_recs)
        
        # Performance tuning
        tuning_recs = self._recommend_performance_tuning(schema_info, query_info, context)
        optimizations.extend(tuning_recs)
        
        return {
            'optimizations': optimizations,
            'schema_analysis': schema_info,
            'query_analysis': query_info,
            'estimated_improvement': self._estimate_db_improvement(optimizations),
            'implementation_priority': self._prioritize_db_optimizations(optimizations)
        }
    
    def suggest_caching_strategy(self, component_code: str, context: Optional[Dict] = None) -> Dict[str, Any]:
        """
        Suggest optimal caching strategies based on code analysis
        
        Args:
            component_code: Backend component source code
            context: Optional context (traffic_pattern, data_volatility, etc.)
            
        Returns:
            Caching strategy recommendations
        """
        # Analyze current caching usage
        current_caching = self._analyze_current_caching(component_code)
        
        # Identify cacheable operations
        cacheable_ops = self._identify_cacheable_operations(component_code)
        
        # Determine optimal caching patterns
        strategies = []
        
        # Memory caching recommendations
        if cacheable_ops['frequent_reads']:
            strategies.append({
                'type': 'memory_cache',
                'pattern': 'redis_cache',
                'use_case': 'Frequent read operations',
                'implementation': self._generate_redis_implementation(cacheable_ops['frequent_reads']),
                'benefits': ['Reduced database load', 'Faster response times'],
                'ttl_recommendation': self._recommend_ttl(cacheable_ops['frequent_reads'])
            })
        
        # CDN recommendations
        if cacheable_ops['static_content']:
            strategies.append({
                'type': 'cdn_cache',
                'pattern': 'edge_caching',
                'use_case': 'Static content delivery',
                'implementation': self._generate_cdn_implementation(cacheable_ops['static_content']),
                'benefits': ['Reduced latency', 'Lower bandwidth costs']
            })
        
        # Application-level caching
        if cacheable_ops['computed_results']:
            strategies.append({
                'type': 'application_cache',
                'pattern': 'memoization',
                'use_case': 'Expensive computations',
                'implementation': self._generate_app_cache_implementation(cacheable_ops['computed_results']),
                'benefits': ['CPU optimization', 'Improved throughput']
            })
        
        return {
            'current_caching': current_caching,
            'cacheable_operations': cacheable_ops,
            'recommended_strategies': strategies,
            'implementation_order': self._prioritize_caching_strategies(strategies),
            'cache_invalidation_strategy': self._recommend_invalidation_strategy(strategies)
        }
    
    def assess_scaling_potential(self, component_code: str, context: Optional[Dict] = None) -> Dict[str, Any]:
        """
        Assess horizontal and vertical scaling potential
        
        Args:
            component_code: Backend component source code
            context: Optional context (current_load, growth_projections, etc.)
            
        Returns:
            Scaling assessment and recommendations
        """
        # Analyze current architecture
        arch_analysis = self._analyze_architecture_patterns(component_code)
        
        # Identify scaling bottlenecks
        bottlenecks = self._identify_scaling_bottlenecks(component_code, arch_analysis)
        
        # Horizontal scaling assessment
        horizontal_scaling = self._assess_horizontal_scaling(component_code, arch_analysis)
        
        # Vertical scaling assessment
        vertical_scaling = self._assess_vertical_scaling(component_code, arch_analysis)
        
        # Microservices decomposition potential
        microservices_potential = self._assess_microservices_potential(component_code)
        
        return {
            'architecture_analysis': arch_analysis,
            'bottlenecks': bottlenecks,
            'horizontal_scaling': horizontal_scaling,
            'vertical_scaling': vertical_scaling,
            'microservices_potential': microservices_potential,
            'scaling_strategy': self._recommend_scaling_strategy(horizontal_scaling, vertical_scaling),
            'resource_requirements': self._estimate_resource_requirements(context)
        }
    
    def suggest_improvements(self, component_code: str, issues: Optional[List] = None) -> Dict[str, Any]:
        """
        Generate specific improvement suggestions for backend components
        
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
            'api_improvements': self._suggest_api_improvements(component_code, issues),
            'database_improvements': self._suggest_db_improvements(component_code, issues),
            'caching_improvements': self._suggest_caching_improvements(component_code, issues),
            'scaling_improvements': self._suggest_scaling_improvements(component_code, issues),
            'security_improvements': self._suggest_security_improvements(component_code, issues),
            'performance_improvements': self._suggest_performance_improvements(component_code, issues)
        }
        
        # Add priority and effort estimates
        prioritized_suggestions = self._prioritize_backend_suggestions(suggestions)
        
        return {
            'suggestions': suggestions,
            'prioritized': prioritized_suggestions,
            'implementation_roadmap': self._create_implementation_roadmap(prioritized_suggestions),
            'estimated_impact': self._estimate_improvement_impact(suggestions)
        }
    
    # REST Compliance Methods
    def check_rest_compliance(self, api_spec: Union[str, Dict]) -> Dict[str, Any]:
        """Check REST API compliance"""
        issues = []
        compliance_score = 100
        
        if isinstance(api_spec, str):
            # Code-based analysis
            # Check HTTP methods usage
            if not re.search(r'(GET|POST|PUT|DELETE|PATCH)', api_spec, re.IGNORECASE):
                issues.append({
                    'type': 'rest_compliance',
                    'severity': 'medium',
                    'message': 'HTTP methods not clearly defined',
                    'line': 1
                })
                compliance_score -= 20
            
            # Check resource naming
            if re.search(r'/api/v\d+/\w+/\w+Action', api_spec):
                issues.append({
                    'type': 'rest_compliance', 
                    'severity': 'medium',
                    'message': 'Non-RESTful endpoint naming detected (action-based)',
                    'line': self._find_issue_line(api_spec, 'Action')
                })
                compliance_score -= 15
            
            # Check status codes
            if not re.search(r'(200|201|400|401|403|404|500)', api_spec):
                issues.append({
                    'type': 'rest_compliance',
                    'severity': 'high',
                    'message': 'HTTP status codes not properly handled',
                    'line': 1
                })
                compliance_score -= 25
        
        return {
            'compliance_score': max(compliance_score, 0),
            'issues': issues,
            'recommendations': self._generate_rest_recommendations(issues)
        }
    
    def analyze_performance(self, api_spec: Union[str, Dict], context: Optional[Dict] = None) -> Dict[str, Any]:
        """Analyze API performance characteristics"""
        issues = []
        performance_score = 100
        
        if isinstance(api_spec, str):
            # Check for N+1 query problems
            n_plus_one = re.findall(r'for.*in.*\n.*query\(|for.*in.*\n.*find\(', api_spec, re.IGNORECASE)
            if n_plus_one:
                issues.append({
                    'type': 'performance',
                    'severity': 'high',
                    'message': f'Potential N+1 query problem detected ({len(n_plus_one)} instances)',
                    'line': self._find_issue_line(api_spec, 'for')
                })
                performance_score -= 30
            
            # Check for missing pagination
            if re.search(r'get.*all|find.*all|select.*from', api_spec, re.IGNORECASE):
                if not re.search(r'limit|offset|page|pagination', api_spec, re.IGNORECASE):
                    issues.append({
                        'type': 'performance',
                        'severity': 'medium',
                        'message': 'Missing pagination for bulk data operations',
                        'line': self._find_issue_line(api_spec, 'all')
                    })
                    performance_score -= 20
            
            # Check for inefficient queries
            if re.search(r'SELECT \*|select \*', api_spec):
                issues.append({
                    'type': 'performance',
                    'severity': 'medium',
                    'message': 'SELECT * queries detected - specify required columns',
                    'line': self._find_issue_line(api_spec, 'SELECT *')
                })
                performance_score -= 15
        
        return {
            'performance_score': max(performance_score, 0),
            'issues': issues,
            'optimization_suggestions': self._generate_performance_optimizations(issues)
        }
    
    def check_api_security(self, api_spec: Union[str, Dict]) -> Dict[str, Any]:
        """Check API security best practices"""
        issues = []
        security_score = 100
        
        if isinstance(api_spec, str):
            # Check for authentication
            if not re.search(r'auth|token|jwt|bearer|oauth', api_spec, re.IGNORECASE):
                issues.append({
                    'type': 'security',
                    'severity': 'high',
                    'message': 'No authentication mechanism detected',
                    'line': 1
                })
                security_score -= 35
            
            # Check for input validation
            if not re.search(r'validate|sanitize|escape|filter', api_spec, re.IGNORECASE):
                issues.append({
                    'type': 'security',
                    'severity': 'high',
                    'message': 'Input validation not detected',
                    'line': 1
                })
                security_score -= 30
            
            # Check for SQL injection protection
            if re.search(r'["\'][^"\']*["\'][^)]*\+[^)]*str\(|\+\s*str\(|["\'][^"\']*WHERE[^"\']*["\'][^)]*\+', api_spec, re.IGNORECASE):
                issues.append({
                    'type': 'security',
                    'severity': 'critical',
                    'message': 'Potential SQL injection vulnerability (string concatenation)',
                    'line': self._find_issue_line(api_spec, '+')
                })
                security_score -= 40
            
            # Check for rate limiting
            if not re.search(r'rate.?limit|throttl|quota', api_spec, re.IGNORECASE):
                issues.append({
                    'type': 'security',
                    'severity': 'medium',
                    'message': 'Rate limiting not implemented',
                    'line': 1
                })
                security_score -= 15
        
        return {
            'security_score': max(security_score, 0),
            'issues': issues,
            'security_recommendations': self._generate_security_recommendations(issues)
        }
    
    def assess_scalability(self, api_spec: Union[str, Dict], context: Optional[Dict] = None) -> Dict[str, Any]:
        """Assess API scalability characteristics"""
        issues = []
        scalability_score = 100
        
        if isinstance(api_spec, str):
            # Check for stateful operations
            if re.search(r'session\[|global |static ', api_spec):
                issues.append({
                    'type': 'scalability',
                    'severity': 'medium',
                    'message': 'Stateful operations detected - may hinder horizontal scaling',
                    'line': self._find_issue_line(api_spec, 'session')
                })
                scalability_score -= 20
            
            # Check for database connections
            if not re.search(r'connection[_\s]*pool|pool[_\s]*connection|\bpool\b', api_spec, re.IGNORECASE):
                if re.search(r'connect|database|db\.|cursor', api_spec, re.IGNORECASE):
                    issues.append({
                        'type': 'scalability',
                        'severity': 'medium',
                        'message': 'Database connection pooling not detected',
                        'line': self._find_issue_line(api_spec, 'connect')
                    })
                    scalability_score -= 25
            
            # Check for caching
            if not re.search(r'cache|redis|memcached', api_spec, re.IGNORECASE):
                issues.append({
                    'type': 'scalability',
                    'severity': 'low',
                    'message': 'No caching mechanism detected',
                    'line': 1
                })
                scalability_score -= 10
        
        return {
            'scalability_score': max(scalability_score, 0),
            'issues': issues,
            'scaling_recommendations': self._generate_scaling_recommendations(issues)
        }
    
    # Database Optimization Methods
    def recommend_indexes(self, schema: Dict, queries: Dict, context: Optional[Dict] = None) -> List[Dict]:
        """Recommend database indexes based on schema and query patterns"""
        recommendations = []
        
        # Analyze WHERE clauses in queries
        for query_info in queries.get('parsed_queries', []):
            where_columns = query_info.get('where_columns', [])
            table = query_info.get('table')
            
            for column in where_columns:
                recommendations.append({
                    'type': 'index_recommendation',
                    'priority': 'high',
                    'table': table,
                    'columns': [column],
                    'index_type': 'btree',
                    'reason': f'Frequent WHERE clause on {column}',
                    'sql': f'CREATE INDEX idx_{table}_{column} ON {table}({column});',
                    'estimated_improvement': '30-60% query speedup'
                })
        
        # Analyze JOIN patterns
        for query_info in queries.get('parsed_queries', []):
            join_columns = query_info.get('join_columns', [])
            
            for join_info in join_columns:
                recommendations.append({
                    'type': 'index_recommendation',
                    'priority': 'high',
                    'table': join_info['table'],
                    'columns': [join_info['column']],
                    'index_type': 'btree',
                    'reason': f'JOIN optimization on {join_info["column"]}',
                    'sql': f'CREATE INDEX idx_{join_info["table"]}_{join_info["column"]} ON {join_info["table"]}({join_info["column"]});',
                    'estimated_improvement': '40-70% JOIN speedup'
                })
        
        return recommendations
    
    def optimize_queries(self, queries: Dict, schema: Dict, context: Optional[Dict] = None) -> List[Dict]:
        """Optimize SQL queries for better performance"""
        optimizations = []
        
        for query_info in queries.get('parsed_queries', []):
            original_query = query_info.get('original_query', '')
            
            # Check for SELECT *
            if 'SELECT *' in original_query.upper():
                optimizations.append({
                    'type': 'query_optimization',
                    'priority': 'medium',
                    'issue': 'SELECT * usage',
                    'original_query': original_query,
                    'optimized_query': self._optimize_select_star(original_query, schema),
                    'reason': 'Specify only required columns to reduce I/O',
                    'estimated_improvement': '10-30% performance gain'
                })
            
            # Check for missing LIMIT clauses
            if 'SELECT' in original_query.upper() and 'LIMIT' not in original_query.upper():
                optimizations.append({
                    'type': 'query_optimization',
                    'priority': 'high',
                    'issue': 'Missing LIMIT clause',
                    'original_query': original_query,
                    'optimized_query': original_query + ' LIMIT 1000',
                    'reason': 'Prevent accidental full table scans',
                    'estimated_improvement': 'Prevents timeout issues'
                })
            
            # Check for inefficient JOINs
            if query_info.get('join_count', 0) > 3:
                optimizations.append({
                    'type': 'query_optimization',
                    'priority': 'medium',
                    'issue': 'Complex JOIN structure',
                    'original_query': original_query,
                    'reason': 'Consider denormalization or query splitting',
                    'recommendation': 'Break into multiple simpler queries or use materialized views'
                })
        
        return optimizations
    
    def suggest_schema_improvements(self, schema: Dict, context: Optional[Dict] = None) -> List[Dict]:
        """Suggest database schema improvements"""
        improvements = []
        
        for table_name, table_info in schema.get('tables', {}).items():
            columns = table_info.get('columns', [])
            
            # Check for missing primary keys
            if not any(col.get('primary_key') for col in columns):
                improvements.append({
                    'type': 'schema_improvement',
                    'priority': 'critical',
                    'table': table_name,
                    'issue': 'Missing primary key',
                    'recommendation': 'Add an auto-incrementing primary key column',
                    'sql': f'ALTER TABLE {table_name} ADD COLUMN id SERIAL PRIMARY KEY;'
                })
            
            # Check for missing NOT NULL constraints
            nullable_columns = [col['name'] for col in columns if col.get('nullable', True)]
            if len(nullable_columns) > len(columns) * 0.7:  # More than 70% nullable
                improvements.append({
                    'type': 'schema_improvement',
                    'priority': 'medium',
                    'table': table_name,
                    'issue': 'Too many nullable columns',
                    'recommendation': 'Add NOT NULL constraints where appropriate',
                    'columns': nullable_columns[:5]  # Show first 5
                })
            
            # Check for text columns without length limits
            unlimited_text = [col['name'] for col in columns 
                            if col.get('type', '').upper() in ['TEXT', 'VARCHAR'] and not col.get('length')]
            if unlimited_text:
                improvements.append({
                    'type': 'schema_improvement',
                    'priority': 'medium',
                    'table': table_name,
                    'issue': 'Unlimited text columns',
                    'recommendation': 'Add appropriate length constraints',
                    'columns': unlimited_text
                })
        
        return improvements
    
    # Helper Methods
    def _identify_backend_type(self, code: str, context: Optional[Dict]) -> Dict[str, Any]:
        """Identify backend technology and patterns"""
        info = {
            'framework': 'unknown',
            'language': 'unknown',
            'database_type': 'unknown',
            'has_api': False,
            'has_database': False,
            'has_caching': False,
            'architecture_pattern': 'unknown'
        }
        
        # Language detection
        if re.search(r'import.*flask|from flask|@app\.route', code, re.IGNORECASE):
            info['framework'] = 'flask'
            info['language'] = 'python'
        elif re.search(r'import.*django|from django', code, re.IGNORECASE):
            info['framework'] = 'django'
            info['language'] = 'python'
        elif re.search(r'import.*express|require.*express', code, re.IGNORECASE):
            info['framework'] = 'express'
            info['language'] = 'javascript'
        elif re.search(r'import.*spring|@RestController|@Service', code, re.IGNORECASE):
            info['framework'] = 'spring'
            info['language'] = 'java'
        
        # Database detection
        if re.search(r'postgresql|postgres|psycopg', code, re.IGNORECASE):
            info['database_type'] = 'postgresql'
        elif re.search(r'mysql|pymysql', code, re.IGNORECASE):
            info['database_type'] = 'mysql'
        elif re.search(r'mongodb|pymongo', code, re.IGNORECASE):
            info['database_type'] = 'mongodb'
        
        # Pattern detection
        info['has_api'] = bool(re.search(r'@.*route|@.*mapping|app\.|router\.|endpoint', code, re.IGNORECASE))
        info['has_database'] = bool(re.search(r'query|cursor|collection|model\.|db\.|session\.', code, re.IGNORECASE))
        info['has_caching'] = bool(re.search(r'cache|redis|memcached', code, re.IGNORECASE))
        
        # Architecture pattern analysis
        arch_analysis = self._analyze_architecture_patterns(code)
        info['architecture_pattern'] = arch_analysis['pattern']
        
        return info
    
    def _calculate_backend_metrics(self, code: str, component_info: Dict) -> Dict[str, Any]:
        """Calculate backend-specific metrics"""
        lines = code.split('\n')
        
        return {
            'lines_of_code': len(lines),
            'cyclomatic_complexity': self._calculate_cyclomatic_complexity(code),
            'api_endpoints': len(re.findall(r'@.*route|@.*mapping|app\.|router\.', code, re.IGNORECASE)),
            'database_queries': len(re.findall(r'query|find|select|insert|update|delete', code, re.IGNORECASE)),
            'external_dependencies': len(re.findall(r'import |from |require\(', code)),
            'error_handling_blocks': len(re.findall(r'try:|except:|catch\s*\(', code)),
            'comment_ratio': len([line for line in lines if line.strip().startswith('#') or line.strip().startswith('//')]) / max(len(lines), 1)
        }
    
    def _calculate_cyclomatic_complexity(self, code: str) -> int:
        """Calculate cyclomatic complexity for backend code"""
        complexity = 1  # Base complexity
        
        # Count decision points
        complexity += len(re.findall(r'\bif\b|\belse\b|\bwhile\b|\bfor\b|\bswitch\b|\bcase\b|\bcatch\b|\?\s*:', code))
        complexity += len(re.findall(r'&&|\|\||and\s|or\s', code))
        
        return complexity
    
    def _calculate_confidence_score(self, issues: List, metrics: Dict) -> float:
        """Calculate confidence score for backend analysis"""
        base_confidence = 0.90
        
        # Reduce confidence based on complexity
        complexity_penalty = min(metrics.get('cyclomatic_complexity', 0) * 0.02, 0.2)
        size_penalty = min(metrics.get('lines_of_code', 0) / 2000, 0.15)
        
        # Increase confidence if we have good coverage
        if metrics.get('api_endpoints', 0) > 0:
            base_confidence += 0.05
        if metrics.get('database_queries', 0) > 0:
            base_confidence += 0.05
        
        confidence = base_confidence - complexity_penalty - size_penalty
        return max(confidence, 0.6)
    
    def _find_issue_line(self, code: str, pattern: str) -> int:
        """Find the line number where an issue occurs"""
        lines = code.split('\n')
        for i, line in enumerate(lines):
            if pattern.lower() in line.lower():
                return i + 1
        return 1
    
    # Pattern Loading Methods
    def _load_api_patterns(self) -> Dict[str, Any]:
        """Load API design patterns"""
        return {
            'rest_principles': {
                'resource_based_urls': 'URLs should represent resources, not actions',
                'http_methods': 'Use appropriate HTTP methods (GET, POST, PUT, DELETE)',
                'stateless': 'Each request should contain all necessary information',
                'cacheable': 'Responses should be cacheable when appropriate'
            },
            'common_issues': {
                'verb_in_url': 'Avoid verbs in URLs (/api/users/getUser -> /api/users/{id})',
                'non_standard_methods': 'Use standard HTTP methods consistently',
                'missing_status_codes': 'Return appropriate HTTP status codes'
            }
        }
    
    def _load_database_patterns(self) -> Dict[str, Any]:
        """Load database optimization patterns"""
        return {
            'indexing_strategies': {
                'single_column': 'For simple WHERE clauses',
                'composite': 'For multi-column queries',
                'partial': 'For filtered indexes',
                'covering': 'To avoid table lookups'
            },
            'query_patterns': {
                'avoid_n_plus_1': 'Use JOINs or batch queries instead of loops',
                'select_specific': 'Avoid SELECT * queries',
                'use_limits': 'Always limit result sets appropriately'
            }
        }
    
    def _load_caching_patterns(self) -> Dict[str, Any]:
        """Load caching strategy patterns"""
        return {
            'cache_types': {
                'memory_cache': 'Redis, Memcached for fast access',
                'database_cache': 'Query result caching',
                'application_cache': 'In-memory caching within application',
                'cdn_cache': 'Content delivery network caching'
            },
            'invalidation_strategies': {
                'ttl_based': 'Time-to-live expiration',
                'event_based': 'Invalidate on data changes',
                'manual': 'Explicit cache clearing'
            }
        }
    
    def _load_scaling_patterns(self) -> Dict[str, Any]:
        """Load scaling patterns"""
        return {
            'horizontal_scaling': {
                'load_balancing': 'Distribute requests across multiple instances',
                'stateless_design': 'Ensure applications are stateless',
                'database_sharding': 'Distribute database load'
            },
            'vertical_scaling': {
                'resource_optimization': 'Optimize CPU and memory usage',
                'connection_pooling': 'Efficient database connections',
                'caching': 'Reduce computational overhead'
            }
        }
    
    # Placeholder methods for comprehensive functionality
    # These would be implemented with full logic in production
    
    def _extract_api_info_from_code(self, code: str) -> Dict:
        """Extract API information from source code"""
        return {'endpoints': [], 'methods': [], 'models': []}
    
    def _check_api_documentation(self, api_spec: Union[str, Dict]) -> Dict:
        """Check API documentation completeness"""
        return {'issues': [], 'documentation_score': 85}
    
    def _check_api_versioning(self, api_spec: Union[str, Dict]) -> Dict:
        """Check API versioning strategy"""
        return {'issues': [], 'versioning_score': 90}
    
    def _check_error_handling(self, api_spec: Union[str, Dict]) -> Dict:
        """Check error handling patterns"""
        return {'issues': [], 'error_handling_score': 80}
    
    def _calculate_api_score(self, analysis_results: Dict) -> float:
        """Calculate overall API quality score"""
        scores = []
        for category, results in analysis_results.items():
            if isinstance(results, dict) and 'score' in results:
                scores.append(results['score'])
        return sum(scores) / len(scores) if scores else 75.0
    
    def _generate_api_recommendations(self, analysis_results: Dict) -> List[Dict]:
        """Generate API improvement recommendations"""
        return []
    
    def _parse_database_schema(self, schema: Union[str, Dict]) -> Dict:
        """Parse database schema"""
        return {'tables': {}, 'relationships': []}
    
    def _parse_queries(self, queries: List[str]) -> Dict:
        """Parse SQL queries"""
        return {'parsed_queries': [], 'patterns': {}}
    
    # Additional helper methods would be implemented here...
    
    def analyze_database_usage(self, code: str, component_info: Dict) -> Dict:
        """Analyze database usage patterns"""
        return {'issues': [], 'patterns': {}}
    
    def analyze_caching_strategies(self, code: str, component_info: Dict) -> Dict:
        """Analyze current caching strategies"""
        return {'issues': [], 'current_strategy': 'none'}
    
    def analyze_scaling_patterns(self, code: str, component_info: Dict) -> Dict:
        """Analyze scaling patterns in the code"""
        return {'issues': [], 'scalability_score': 75}
    
    def _check_security_patterns(self, code: str, component_info: Dict) -> List[Dict]:
        """Check security patterns"""
        return []
    
    def _check_performance_patterns(self, code: str, component_info: Dict) -> List[Dict]:
        """Check performance patterns"""
        return []
    
    def _generate_priority_recommendations(self, issues: List) -> List[Dict]:
        """Generate prioritized recommendations"""
        return []
    
    def _suggest_api_improvements(self, code: str, issues: List) -> List[Dict]:
        """Suggest API improvements"""
        return []
    
    def _suggest_db_improvements(self, code: str, issues: List) -> List[Dict]:
        """Suggest database improvements"""
        return []
    
    def _suggest_caching_improvements(self, code: str, issues: List) -> List[Dict]:
        """Suggest caching improvements"""
        return []
    
    def _suggest_scaling_improvements(self, code: str, issues: List) -> List[Dict]:
        """Suggest scaling improvements"""
        return []
    
    def _suggest_security_improvements(self, code: str, issues: List) -> List[Dict]:
        """Suggest security improvements"""
        return []
    
    def _suggest_performance_improvements(self, code: str, issues: List) -> List[Dict]:
        """Suggest performance improvements"""
        return []
    
    def _prioritize_backend_suggestions(self, suggestions: Dict) -> List[Dict]:
        """Prioritize backend suggestions"""
        return []
    
    def _create_implementation_roadmap(self, prioritized_suggestions: List) -> Dict:
        """Create implementation roadmap"""
        return {'phases': [], 'timeline': ''}
    
    def _estimate_improvement_impact(self, suggestions: Dict) -> Dict:
        """Estimate improvement impact"""
        return {'performance': 'medium', 'scalability': 'high'}

# Missing helper methods implementation

    def _generate_rest_recommendations(self, issues: List[Dict]) -> List[Dict]:
        """Generate REST compliance recommendations"""
        recommendations = []
        for issue in issues:
            if 'HTTP methods not clearly defined' in issue.get('message', ''):
                recommendations.append({
                    'type': 'rest_compliance',
                    'priority': 'medium',
                    'suggestion': 'Use explicit HTTP methods (GET, POST, PUT, DELETE)',
                    'example': '@app.route("/api/users", methods=["GET", "POST"])'
                })
            elif 'status codes' in issue.get('message', ''):
                recommendations.append({
                    'type': 'rest_compliance',
                    'priority': 'high',
                    'suggestion': 'Return appropriate HTTP status codes',
                    'example': 'return jsonify(data), 200  # or 201, 404, etc.'
                })
        return recommendations

    def _generate_performance_optimizations(self, issues: List[Dict]) -> List[Dict]:
        """Generate performance optimization suggestions"""
        optimizations = []
        for issue in issues:
            if 'N+1' in issue.get('message', ''):
                optimizations.append({
                    'type': 'performance',
                    'priority': 'high',
                    'suggestion': 'Use JOIN queries instead of loops',
                    'example': 'SELECT * FROM posts p JOIN users u ON p.user_id = u.id'
                })
            elif 'pagination' in issue.get('message', ''):
                optimizations.append({
                    'type': 'performance',
                    'priority': 'medium',
                    'suggestion': 'Add pagination to bulk operations',
                    'example': 'LIMIT 20 OFFSET ?'
                })
        return optimizations

    def _generate_security_recommendations(self, issues: List[Dict]) -> List[Dict]:
        """Generate security recommendations"""
        recommendations = []
        for issue in issues:
            if 'SQL injection' in issue.get('message', ''):
                recommendations.append({
                    'type': 'security',
                    'priority': 'critical',
                    'suggestion': 'Use parameterized queries',
                    'example': 'cursor.execute("SELECT * FROM users WHERE id = ?", (user_id,))'
                })
            elif 'authentication' in issue.get('message', ''):
                recommendations.append({
                    'type': 'security',
                    'priority': 'high',
                    'suggestion': 'Implement proper authentication',
                    'example': 'Use JWT tokens or OAuth'
                })
        return recommendations

    def _generate_scaling_recommendations(self, issues: List[Dict]) -> List[Dict]:
        """Generate scaling recommendations"""
        recommendations = []
        for issue in issues:
            if 'stateful' in issue.get('message', ''):
                recommendations.append({
                    'type': 'scaling',
                    'priority': 'medium',
                    'suggestion': 'Design stateless components',
                    'example': 'Use external session storage (Redis)'
                })
            elif 'connection' in issue.get('message', ''):
                recommendations.append({
                    'type': 'scaling',
                    'priority': 'high',
                    'suggestion': 'Implement connection pooling',
                    'example': 'Use SQLAlchemy connection pool'
                })
        return recommendations

    def _recommend_partitioning(self, schema_info: Dict, context: Optional[Dict] = None) -> List[Dict]:
        """Recommend database partitioning strategies"""
        return []

    def _recommend_performance_tuning(self, schema_info: Dict, query_info: Dict, context: Optional[Dict] = None) -> List[Dict]:
        """Recommend performance tuning options"""
        return []

    def _estimate_db_improvement(self, optimizations: List[Dict]) -> Dict[str, Any]:
        """Estimate database improvement potential"""
        if not optimizations:
            return {'performance_gain': '0%', 'confidence': 'low'}
        
        high_priority = len([opt for opt in optimizations if opt.get('priority') == 'high'])
        medium_priority = len([opt for opt in optimizations if opt.get('priority') == 'medium'])
        
        estimated_gain = high_priority * 30 + medium_priority * 15  # Rough percentage
        
        return {
            'performance_gain': f'{min(estimated_gain, 80)}%',
            'confidence': 'high' if high_priority > 0 else 'medium'
        }

    def _prioritize_db_optimizations(self, optimizations: List[Dict]) -> List[Dict]:
        """Prioritize database optimizations"""
        priority_order = {'critical': 4, 'high': 3, 'medium': 2, 'low': 1}
        return sorted(optimizations, key=lambda x: priority_order.get(x.get('priority', 'low'), 0), reverse=True)

    def _analyze_current_caching(self, component_code: str) -> Dict[str, Any]:
        """Analyze current caching implementation"""
        has_redis = bool(re.search(r'redis|Redis', component_code))
        has_memcached = bool(re.search(r'memcached|Memcached', component_code))
        has_cache_headers = bool(re.search(r'Cache-Control|Expires', component_code))
        
        return {
            'cache_types': {
                'redis': has_redis,
                'memcached': has_memcached,
                'http_headers': has_cache_headers
            },
            'cache_usage': 'present' if (has_redis or has_memcached) else 'none'
        }

    def _identify_cacheable_operations(self, component_code: str) -> Dict[str, List]:
        """Identify operations that could benefit from caching"""
        frequent_reads = []
        static_content = []
        computed_results = []
        
        # Look for database SELECT operations
        db_reads = re.findall(r'SELECT.*FROM.*', component_code, re.IGNORECASE)
        if db_reads:
            frequent_reads.extend(['Database queries detected'])
        
        # Look for static file serving
        static_patterns = re.findall(r'\.(js|css|png|jpg|gif)', component_code, re.IGNORECASE)
        if static_patterns:
            static_content.extend(['Static files detected'])
        
        # Look for expensive computations
        compute_patterns = re.findall(r'(sum|count|max|min)\s*\(', component_code, re.IGNORECASE)
        if compute_patterns:
            computed_results.extend(['Computational operations detected'])
        
        return {
            'frequent_reads': frequent_reads,
            'static_content': static_content,
            'computed_results': computed_results
        }

    def _generate_redis_implementation(self, operations: List) -> str:
        """Generate Redis implementation example"""
        return """
import redis
redis_client = redis.Redis(host='localhost')

# Cache implementation
cache_key = f"data:{id}"
cached = redis_client.get(cache_key)
if cached:
    return json.loads(cached)
    
# Fetch data and cache
data = fetch_from_db(id)
redis_client.setex(cache_key, 300, json.dumps(data))  # 5 min TTL
        """

    def _generate_cdn_implementation(self, operations: List) -> str:
        """Generate CDN implementation example"""
        return """
# Configure CDN headers
@app.after_request
def add_cache_headers(response):
    if request.endpoint == 'static':
        response.cache_control.max_age = 3600  # 1 hour
    return response
        """

    def _generate_app_cache_implementation(self, operations: List) -> str:
        """Generate application cache implementation example"""
        return """
from functools import lru_cache

@lru_cache(maxsize=128)
def expensive_computation(param):
    # Cached computation
    return compute_result(param)
        """

    def _recommend_ttl(self, operations: List) -> Dict[str, int]:
        """Recommend TTL values for different operations"""
        return {
            'user_data': 300,      # 5 minutes
            'static_content': 3600, # 1 hour  
            'computed_results': 900 # 15 minutes
        }

    def _prioritize_caching_strategies(self, strategies: List[Dict]) -> List[Dict]:
        """Prioritize caching strategies by impact"""
        priority_map = {'memory_cache': 3, 'application_cache': 2, 'cdn_cache': 1}
        return sorted(strategies, key=lambda x: priority_map.get(x.get('type', ''), 0), reverse=True)

    def _recommend_invalidation_strategy(self, strategies: List[Dict]) -> Dict[str, Any]:
        """Recommend cache invalidation strategy"""
        return {
            'strategy': 'event_based',
            'description': 'Invalidate cache when underlying data changes',
            'implementation': 'Use Redis pub/sub or database triggers'
        }

    def _analyze_architecture_patterns(self, component_code: str) -> Dict[str, Any]:
        """Analyze current architecture patterns"""
        # Detect microservices patterns
        has_circuit_breaker = bool(re.search(r'circuit|@circuit', component_code, re.IGNORECASE))
        has_service_calls = bool(re.search(r'requests\.get|http[s]?://.*service', component_code, re.IGNORECASE))
        has_distributed_patterns = has_circuit_breaker or has_service_calls
        
        endpoints = len(re.findall(r'@app\.route', component_code))
        
        if has_distributed_patterns:
            pattern = 'microservices'
        elif endpoints > 10:
            pattern = 'monolithic'
        else:
            pattern = 'modular'
            
        return {
            'pattern': pattern,
            'database_access': 'direct' if 'sqlite3.connect' in component_code else 'abstracted',
            'stateless': 'session[' not in component_code,
            'microservices_indicators': has_distributed_patterns
        }

    def _identify_scaling_bottlenecks(self, component_code: str, arch_analysis: Dict) -> List[str]:
        """Identify scaling bottlenecks"""
        bottlenecks = []
        
        if 'session[' in component_code:
            bottlenecks.append('Stateful session usage')
        
        if not re.search(r'pool|Pool', component_code):
            bottlenecks.append('No connection pooling detected')
            
        if re.search(r'for.*in.*:\s*.*query', component_code, re.DOTALL):
            bottlenecks.append('N+1 query pattern detected')
        
        return bottlenecks

    def _assess_horizontal_scaling(self, component_code: str, arch_analysis: Dict) -> Dict[str, Any]:
        """Assess horizontal scaling potential"""
        stateless = arch_analysis.get('stateless', False)
        
        return {
            'feasible': stateless,
            'blockers': [] if stateless else ['Stateful components detected'],
            'recommendations': ['Use load balancers', 'Implement session storage']
        }

    def _assess_vertical_scaling(self, component_code: str, arch_analysis: Dict) -> Dict[str, Any]:
        """Assess vertical scaling potential"""
        return {
            'cpu_intensive': bool(re.search(r'for.*range|while.*:', component_code)),
            'memory_intensive': bool(re.search(r'\.fetchall\(\)|SELECT \*', component_code)),
            'recommendations': ['Optimize algorithms', 'Use pagination', 'Add indexing']
        }

    def _assess_microservices_potential(self, component_code: str) -> Dict[str, Any]:
        """Assess potential for microservices decomposition"""
        endpoints = len(re.findall(r'@app\.route', component_code))
        
        return {
            'suitable': endpoints > 5,
            'decomposition_strategy': 'domain-based' if endpoints > 5 else 'not_recommended',
            'estimated_services': max(1, endpoints // 5)
        }

    def _recommend_scaling_strategy(self, horizontal: Dict, vertical: Dict) -> Dict[str, Any]:
        """Recommend optimal scaling strategy"""
        if horizontal.get('feasible', False):
            return {
                'primary': 'horizontal',
                'secondary': 'vertical',
                'reasoning': 'Application is stateless and suitable for horizontal scaling'
            }
        else:
            return {
                'primary': 'vertical',
                'secondary': 'architecture_refactor',
                'reasoning': 'Address stateful components before horizontal scaling'
            }

    def _estimate_resource_requirements(self, context: Optional[Dict]) -> Dict[str, Any]:
        """Estimate resource requirements for scaling"""
        if not context:
            context = {}
        
        current_load = context.get('current_load', 'unknown')
        growth_rate = context.get('growth_rate', 'moderate')
        
        return {
            'cpu_recommendation': '2-4 cores per instance',
            'memory_recommendation': '2-4 GB per instance',
            'storage_recommendation': 'Scale based on data growth',
            'network_recommendation': 'High bandwidth for API-heavy workloads'
        }

    def _optimize_select_star(self, query: str, schema: Dict) -> str:
        """Optimize SELECT * queries"""
        # Simple replacement for demo
        if 'SELECT *' in query.upper():
            return query.replace('SELECT *', 'SELECT id, name, email')  # Example specific columns
        return query

    def validate_capability(self, capability: str) -> bool:
        """Validate if the agent has a specific capability"""
        return capability in self.capabilities

