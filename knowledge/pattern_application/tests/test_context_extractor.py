"""
Tests for Context Extractor

Comprehensive tests for the ContextExtractor class,
including GitHub issue parsing, technology detection, and complexity assessment.
"""

import pytest
from knowledge.pattern_application.context_extractor import ContextExtractor
from knowledge.pattern_application.core import IssueContext, TechStack, IssueConstraints


class TestContextExtractor:
    """Test suite for ContextExtractor."""
    
    @pytest.fixture
    def extractor(self):
        """Create ContextExtractor instance."""
        return ContextExtractor()
    
    @pytest.fixture
    def sample_github_issue(self):
        """Sample GitHub issue data for testing."""
        return {
            'number': 42,
            'title': 'Implement user authentication API',
            'body': '''
            ## Description
            We need to implement a secure user authentication API using FastAPI and PostgreSQL.
            
            ## Requirements
            - JWT token authentication
            - Password hashing with bcrypt
            - User registration and login endpoints
            - Rate limiting
            - Test coverage > 80%
            
            ## Technical Details
            - Language: Python 3.9+
            - Framework: FastAPI
            - Database: PostgreSQL
            - Deployment: Docker containers on AWS
            - Timeline: 2 weeks
            ''',
            'labels': [
                {'name': 'enhancement'},
                {'name': 'backend'},
                {'name': 'security'}
            ]
        }
    
    @pytest.fixture
    def complex_github_issue(self):
        """Complex GitHub issue for testing high complexity detection."""
        return {
            'number': 123,
            'title': 'Migrate microservices to event-driven architecture',
            'body': '''
            ## Epic: Microservices Architecture Migration
            
            We need to migrate our current monolithic architecture to a scalable, 
            event-driven microservices architecture with real-time processing capabilities.
            
            ## Requirements
            - Implement service mesh with Istio
            - Set up event streaming with Apache Kafka
            - Create distributed tracing with Jaeger
            - Implement circuit breakers
            - Database sharding across services
            - Performance: <100ms response time, >99.9% availability
            - Security: OAuth2, mTLS between services
            - Compliance: GDPR, SOX compliance required
            - Integration with existing legacy systems
            
            ## Technical Stack
            - Languages: Go, Java, Python
            - Frameworks: Spring Boot, Gin, FastAPI
            - Databases: PostgreSQL, MongoDB, Redis
            - Message Queue: Apache Kafka
            - Orchestration: Kubernetes
            - Service Mesh: Istio
            - Monitoring: Prometheus, Grafana, Jaeger
            - Deployment: GitLab CI/CD, Helm charts
            
            ## Constraints
            - Deadline: Q1 2024 (urgent)
            - Budget: $500K maximum
            - Zero downtime migration required
            - Backward compatibility with mobile apps
            
            This is a breaking change that affects the entire system architecture.
            ''',
            'labels': [
                {'name': 'epic'},
                {'name': 'architecture'},
                {'name': 'breaking-change'},
                {'name': 'microservices'},
                {'name': 'high-priority'}
            ]
        }
    
    def test_extract_issue_context_basic(self, extractor, sample_github_issue):
        """Test basic issue context extraction."""
        context = extractor.extract_issue_context(sample_github_issue)
        
        assert isinstance(context, IssueContext)
        assert context.issue_id == '42'
        assert context.title == 'Implement user authentication API'
        assert 'authentication' in context.description
        assert context.labels == ['enhancement', 'backend', 'security']
    
    def test_extract_issue_context_complex(self, extractor, complex_github_issue):
        """Test complex issue context extraction."""
        context = extractor.extract_issue_context(complex_github_issue)
        
        assert context.issue_id == '123'
        assert context.complexity == 'very-high'  # Should detect very high complexity
        assert 'microservices' in context.description.lower()
        assert len(context.labels) == 5
    
    def test_detect_tech_stack_python(self, extractor):
        """Test technology stack detection for Python projects."""
        content = """
        We need to implement a REST API using FastAPI and PostgreSQL.
        The application will be containerized with Docker and deployed on AWS.
        Testing will be done with pytest and we'll use SQLAlchemy as ORM.
        """
        
        tech_stack = extractor.detect_tech_stack(content)
        
        assert tech_stack.primary_language == 'python'
        assert 'fastapi' in tech_stack.frameworks
        assert 'postgresql' in tech_stack.databases
        assert 'docker' in tech_stack.tools
        assert tech_stack.architecture_pattern == 'rest'
        assert tech_stack.deployment_target == 'cloud'
    
    def test_detect_tech_stack_javascript(self, extractor):
        """Test technology stack detection for JavaScript projects."""
        content = """
        Build a React frontend application with Node.js backend.
        Use Express.js for the API, MongoDB for database, and Jest for testing.
        Deploy using Docker containers on Kubernetes.
        """
        
        tech_stack = extractor.detect_tech_stack(content)
        
        assert tech_stack.primary_language == 'javascript'
        assert 'react' in tech_stack.frameworks
        assert 'express' in tech_stack.frameworks
        assert 'mongodb' in tech_stack.databases
        assert 'docker' in tech_stack.tools
        assert 'kubernetes' in tech_stack.tools
    
    def test_detect_tech_stack_java(self, extractor):
        """Test technology stack detection for Java projects."""
        content = """
        Implement microservices using Spring Boot and Maven.
        Database will be MySQL with JPA/Hibernate.
        Testing with JUnit and deployment on traditional servers.
        """
        
        tech_stack = extractor.detect_tech_stack(content)
        
        assert tech_stack.primary_language == 'java'
        assert 'spring boot' in tech_stack.frameworks
        assert 'mysql' in tech_stack.databases
        assert tech_stack.deployment_target == 'traditional'
    
    def test_detect_tech_stack_unknown(self, extractor):
        """Test technology stack detection with unknown/unclear content."""
        content = """
        We need to fix a bug in the system.
        The issue is causing problems for users.
        """
        
        tech_stack = extractor.detect_tech_stack(content)
        
        assert tech_stack.primary_language == 'unknown'
        assert len(tech_stack.frameworks) == 0
        assert len(tech_stack.databases) == 0
    
    def test_assess_complexity_low(self, extractor):
        """Test complexity assessment for low complexity issues."""
        context = IssueContext(
            issue_id='1',
            title='Fix typo in documentation',
            description='There is a typo in the README file that needs to be fixed.',
            complexity='medium',  # Will be overridden by assessment
            tech_stack=TechStack(primary_language='unknown'),
            constraints=IssueConstraints(),
            domain='documentation',
            labels=['documentation', 'typo']
        )
        
        complexity = extractor.assess_complexity(context)
        
        assert complexity == 'low'
    
    def test_assess_complexity_medium(self, extractor):
        """Test complexity assessment for medium complexity issues."""
        context = IssueContext(
            issue_id='2',
            title='Add user authentication',
            description='Implement user authentication with JWT tokens and database integration.',
            complexity='medium',
            tech_stack=TechStack(
                primary_language='python',
                frameworks=['fastapi'],
                databases=['postgresql']
            ),
            constraints=IssueConstraints(
                quality_gates=['unit tests', 'integration tests']
            ),
            domain='backend',
            labels=['enhancement', 'authentication']
        )
        
        complexity = extractor.assess_complexity(context)
        
        assert complexity == 'medium'
    
    def test_assess_complexity_high(self, extractor):
        """Test complexity assessment for high complexity issues."""
        context = IssueContext(
            issue_id='3',
            title='Implement microservices architecture',
            description='Migrate from monolith to microservices with distributed database, '
                       'service discovery, and performance optimization.',
            complexity='medium',
            tech_stack=TechStack(
                primary_language='java',
                frameworks=['spring boot', 'spring cloud'],
                databases=['postgresql', 'mongodb', 'redis'],
                architecture_pattern='microservices'
            ),
            constraints=IssueConstraints(
                performance_requirements={'response_time': '100ms'},
                quality_gates=['unit tests', 'integration tests', 'performance tests']
            ),
            domain='backend',
            labels=['architecture', 'microservices', 'performance']
        )
        
        complexity = extractor.assess_complexity(context)
        
        assert complexity in ['high', 'very-high']
    
    def test_assess_complexity_very_high(self, extractor, complex_github_issue):
        """Test complexity assessment for very high complexity issues."""
        context = extractor.extract_issue_context(complex_github_issue)
        
        assert context.complexity == 'very-high'
    
    def test_extract_constraints_timeline(self, extractor):
        """Test timeline constraint extraction."""
        content = """
        This feature is urgent and needs to be completed by March 15th.
        The deadline is firm due to regulatory requirements.
        """
        
        constraints = extractor.extract_constraints(content)
        
        assert constraints.timeline is not None
        assert 'march 15th' in constraints.timeline.lower() or 'urgent' in constraints.timeline.lower()
    
    def test_extract_constraints_performance(self, extractor):
        """Test performance constraint extraction."""
        content = """
        The API must have a response time of less than 200ms.
        System should handle 1000 requests per second.
        Availability must be 99.9% or higher.
        """
        
        constraints = extractor.extract_constraints(content)
        
        assert constraints.performance_requirements is not None
        assert 'response_time' in constraints.performance_requirements
        assert 'throughput' in constraints.performance_requirements
        assert 'availability' in constraints.performance_requirements
    
    def test_extract_constraints_compliance(self, extractor):
        """Test compliance constraint extraction."""
        content = """
        This system must be GDPR compliant and meet HIPAA requirements.
        PCI DSS compliance is also required for payment processing.
        """
        
        constraints = extractor.extract_constraints(content)
        
        assert 'GDPR' in constraints.compliance_requirements
        assert 'HIPAA' in constraints.compliance_requirements
        assert 'PCI' in constraints.compliance_requirements
    
    def test_extract_constraints_quality_gates(self, extractor):
        """Test quality gate constraint extraction."""
        content = """
        All code must pass unit tests and integration tests.
        Test coverage should be at least 80%.
        Code review is required before merge.
        Security scan must be completed.
        """
        
        constraints = extractor.extract_constraints(content)
        
        assert len(constraints.quality_gates) > 0
        quality_gates_text = ' '.join(constraints.quality_gates).lower()
        assert 'test' in quality_gates_text
        assert 'code review' in quality_gates_text or 'security scan' in quality_gates_text
    
    def test_detect_domain_backend(self, extractor):
        """Test domain detection for backend issues."""
        content = "Implement REST API with database integration and server-side authentication."
        
        domain = extractor._detect_domain(content)
        
        assert domain == 'backend'
    
    def test_detect_domain_frontend(self, extractor):
        """Test domain detection for frontend issues."""
        content = "Create responsive React UI components with CSS animations."
        
        domain = extractor._detect_domain(content)
        
        assert domain == 'frontend'
    
    def test_detect_domain_devops(self, extractor):
        """Test domain detection for DevOps issues."""
        content = "Set up Docker containers with Kubernetes deployment and CI/CD pipeline."
        
        domain = extractor._detect_domain(content)
        
        assert domain == 'devops'
    
    def test_detect_domain_security(self, extractor):
        """Test domain detection for security issues."""
        content = "Implement authentication system with encryption and authorization controls."
        
        domain = extractor._detect_domain(content)
        
        assert domain == 'security'
    
    def test_detect_domain_general(self, extractor):
        """Test domain detection for general issues."""
        content = "Update documentation and fix miscellaneous issues."
        
        domain = extractor._detect_domain(content)
        
        # Should default to general or documentation
        assert domain in ['general', 'documentation']
    
    def test_extract_similar_issues(self, extractor):
        """Test similar issue reference extraction."""
        content = """
        This is similar to issue #123 and relates to #456.
        Also see issue #789 for related work.
        """
        
        similar_issues = extractor._extract_similar_issues(content)
        
        assert '123' in similar_issues
        assert '456' in similar_issues
        assert '789' in similar_issues
        assert len(similar_issues) == 3
    
    def test_extract_pattern_references(self, extractor):
        """Test pattern reference extraction."""
        content = "Use the same pattern as before and follow the template approach."
        labels = ['pattern-match', 'template-based']
        
        pattern_refs = extractor._extract_pattern_references(content, labels)
        
        assert len(pattern_refs) > 0
        assert any('pattern' in ref.lower() for ref in pattern_refs)
    
    def test_language_patterns_comprehensive(self, extractor):
        """Test comprehensive language pattern matching."""
        test_cases = [
            ("Use JavaScript with Node.js and React", 'javascript'),
            ("Python Django application with pytest", 'python'),
            ("Java Spring Boot microservice", 'java'),
            ("Go backend with Gin framework", 'go'),
            ("Rust application with Cargo", 'rust'),
            ("C# .NET Core API", 'csharp'),
            ("PHP Laravel web application", 'php'),
            ("Ruby on Rails application", 'ruby'),
        ]
        
        for content, expected_language in test_cases:
            tech_stack = extractor.detect_tech_stack(content)
            assert tech_stack.primary_language == expected_language, f"Failed for: {content}"
    
    def test_architecture_pattern_detection(self, extractor):
        """Test architecture pattern detection."""
        test_cases = [
            ("RESTful API design", 'rest'),
            ("GraphQL endpoint implementation", 'graphql'),
            ("Microservices architecture", 'microservices'),
            ("Monolithic application", 'monolith'),
            ("Serverless Lambda functions", 'serverless'),
            ("Event-driven system", 'event-driven'),
            ("MVC pattern implementation", 'mvc'),
        ]
        
        for content, expected_pattern in test_cases:
            tech_stack = extractor.detect_tech_stack(content)
            assert tech_stack.architecture_pattern == expected_pattern, f"Failed for: {content}"
    
    def test_deployment_target_detection(self, extractor):
        """Test deployment target detection."""
        test_cases = [
            ("Deploy on AWS cloud", 'cloud'),
            ("Kubernetes cluster deployment", 'kubernetes'),
            ("Docker container deployment", 'docker'),
            ("Lambda serverless functions", 'serverless'),
            ("Traditional server deployment", 'traditional'),
        ]
        
        for content, expected_target in test_cases:
            tech_stack = extractor.detect_tech_stack(content)
            assert tech_stack.deployment_target == expected_target, f"Failed for: {content}"


if __name__ == '__main__':
    pytest.main([__file__, '-v'])