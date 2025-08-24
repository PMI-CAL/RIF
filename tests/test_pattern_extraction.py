"""
Comprehensive test suite for pattern extraction engine components.

Tests all major components of the pattern extraction system including
discovery engine, extractors, and success metrics calculator.
"""

import unittest
import json
import tempfile
import os
from datetime import datetime, timedelta
from unittest.mock import Mock, patch, MagicMock
import sys
from pathlib import Path

# Add knowledge directory to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent / 'knowledge'))

from knowledge.pattern_extraction.discovery_engine import (
    PatternDiscoveryEngine, 
    ExtractedPattern, 
    PatternSignature
)
from knowledge.pattern_extraction.code_extractor import (
    CodePatternExtractor, 
    CodeStructure, 
    ArchitecturalElement
)
from knowledge.pattern_extraction.workflow_extractor import (
    WorkflowPatternExtractor,
    StateTransition,
    WorkflowSequence
)
from knowledge.pattern_extraction.decision_extractor import (
    DecisionPatternExtractor,
    DecisionPattern,
    DecisionContext,
    DecisionAlternative,
    DecisionOutcome
)
from knowledge.pattern_extraction.success_metrics import (
    SuccessMetricsCalculator,
    SuccessMetrics,
    PatternApplication
)


class TestPatternSignature(unittest.TestCase):
    """Test PatternSignature functionality."""
    
    def setUp(self):
        self.pattern_data = {
            'title': 'Test Pattern',
            'description': 'A test pattern for unit testing',
            'architecture': {'layers': ['presentation', 'business', 'data']},
            'components': ['controller', 'service', 'repository'],
            'complexity': 'medium',
            'domain': 'web',
            'tags': ['mvc', 'layered']
        }
    
    def test_signature_generation(self):
        """Test pattern signature generation."""
        signature = PatternSignature.from_pattern(self.pattern_data)
        
        self.assertIsNotNone(signature.content_hash)
        self.assertIsNotNone(signature.structure_hash)
        self.assertIsNotNone(signature.metadata_hash)
        self.assertIsNotNone(signature.combined_hash)
        
        # Hashes should be consistent
        signature2 = PatternSignature.from_pattern(self.pattern_data)
        self.assertEqual(signature.combined_hash, signature2.combined_hash)
    
    def test_signature_uniqueness(self):
        """Test that different patterns produce different signatures."""
        pattern_data2 = self.pattern_data.copy()
        pattern_data2['title'] = 'Different Pattern'
        
        signature1 = PatternSignature.from_pattern(self.pattern_data)
        signature2 = PatternSignature.from_pattern(pattern_data2)
        
        self.assertNotEqual(signature1.combined_hash, signature2.combined_hash)


class TestPatternDiscoveryEngine(unittest.TestCase):
    """Test PatternDiscoveryEngine functionality."""
    
    def setUp(self):
        # Mock knowledge system
        self.mock_knowledge = Mock()
        self.mock_knowledge.store_knowledge.return_value = "pattern_123"
        
        self.engine = PatternDiscoveryEngine(self.mock_knowledge)
        
        # Sample completed issue data
        self.completed_issue = {
            'issue_number': 42,
            'title': 'Implement user authentication',
            'body': 'Add JWT-based authentication system',
            'code_changes': {
                'auth.py': {
                    'added_lines': 'class AuthService:\n    def authenticate(self, token):\n        pass'
                }
            },
            'history': [
                {
                    'timestamp': '2023-01-01T10:00:00Z',
                    'label_added': 'state:new'
                },
                {
                    'timestamp': '2023-01-01T11:00:00Z',
                    'label_added': 'state:implementing'
                }
            ],
            'decisions': [
                {
                    'title': 'Choose JWT for authentication',
                    'context': 'Need secure token-based auth',
                    'decision': 'Use JWT tokens for stateless authentication'
                }
            ]
        }
    
    def test_initialization(self):
        """Test engine initialization."""
        self.assertIsNotNone(self.engine)
        self.assertEqual(len(self.engine.extractors), 0)
        self.assertEqual(len(self.engine.pattern_signatures), 0)
    
    def test_extractor_registration(self):
        """Test extractor registration."""
        mock_extractor = Mock()
        self.engine.register_extractor('test', mock_extractor)
        
        self.assertIn('test', self.engine.extractors)
        self.assertEqual(self.engine.extractors['test'], mock_extractor)
    
    def test_pattern_discovery(self):
        """Test pattern discovery with mock extractors."""
        # Register mock extractor
        mock_extractor = Mock()
        mock_pattern = ExtractedPattern(
            title='Mock Pattern',
            description='Test pattern',
            pattern_type='test',
            source='test',
            content={},
            context={},
            signature=PatternSignature.from_pattern({'title': 'Mock Pattern'}),
            extraction_method='mock',
            confidence=0.8,
            created_at=datetime.now()
        )
        mock_extractor.extract_patterns.return_value = [mock_pattern]
        
        self.engine.register_extractor('test', mock_extractor)
        
        # Discover patterns
        patterns = self.engine.discover_patterns(self.completed_issue)
        
        self.assertEqual(len(patterns), 1)
        self.assertEqual(patterns[0].title, 'Mock Pattern')
        mock_extractor.extract_patterns.assert_called_once_with(self.completed_issue)
    
    def test_deduplication(self):
        """Test pattern deduplication."""
        # Create two identical patterns
        pattern1 = ExtractedPattern(
            title='Duplicate Pattern',
            description='Test',
            pattern_type='test',
            source='test1',
            content={},
            context={},
            signature=PatternSignature.from_pattern({'title': 'Duplicate Pattern'}),
            extraction_method='test',
            confidence=0.8,
            created_at=datetime.now()
        )
        
        pattern2 = ExtractedPattern(
            title='Duplicate Pattern',
            description='Test',
            pattern_type='test',
            source='test2',
            content={},
            context={},
            signature=PatternSignature.from_pattern({'title': 'Duplicate Pattern'}),
            extraction_method='test',
            confidence=0.8,
            created_at=datetime.now()
        )
        
        patterns = [pattern1, pattern2]
        unique_patterns = self.engine._deduplicate_patterns(patterns)
        
        self.assertEqual(len(unique_patterns), 1)
        self.assertEqual(self.engine.extraction_stats['duplicates_filtered'], 1)
    
    def test_pattern_storage(self):
        """Test pattern storage in knowledge base."""
        pattern = ExtractedPattern(
            title='Storage Test Pattern',
            description='Test',
            pattern_type='test',
            source='test',
            content={},
            context={},
            signature=PatternSignature.from_pattern({'title': 'Storage Test Pattern'}),
            extraction_method='test',
            confidence=0.8,
            created_at=datetime.now()
        )
        
        result = self.engine._store_pattern(pattern)
        
        self.assertTrue(result)
        self.mock_knowledge.store_knowledge.assert_called_once()
        
        # Check call arguments
        call_args = self.mock_knowledge.store_knowledge.call_args
        self.assertEqual(call_args[1]['collection'], 'patterns')


class TestCodePatternExtractor(unittest.TestCase):
    """Test CodePatternExtractor functionality."""
    
    def setUp(self):
        self.extractor = CodePatternExtractor()
        
        self.sample_python_code = '''
class UserService:
    def __init__(self, repository):
        self.repository = repository
    
    def create_user(self, user_data):
        try:
            return self.repository.save(user_data)
        except Exception as e:
            logger.error(f"Error creating user: {e}")
            raise
'''
        
        self.completed_issue = {
            'issue_number': 25,
            'code_changes': {
                'user_service.py': {
                    'added_lines': self.sample_python_code
                }
            },
            'decisions': [
                {
                    'title': 'Use repository pattern',
                    'context': 'Need to abstract database access',
                    'decision': 'Implement repository pattern for data access'
                }
            ],
            'files_created': ['user_service.py']
        }
    
    def test_initialization(self):
        """Test extractor initialization."""
        self.assertIsNotNone(self.extractor)
        self.assertIn('singleton', self.extractor.design_patterns)
        self.assertIn('layered_architecture', self.extractor.architectural_patterns)
        self.assertIn('error_handling', self.extractor.quality_patterns)
    
    def test_pattern_extraction(self):
        """Test pattern extraction from completed issue."""
        patterns = self.extractor.extract_patterns(self.completed_issue)
        
        self.assertGreater(len(patterns), 0)
        
        # Check that we get different types of patterns
        pattern_types = {p.pattern_type for p in patterns}
        self.assertIn('decision', pattern_types)  # From decisions
    
    def test_code_structure_extraction(self):
        """Test AST-based code structure extraction."""
        import ast
        tree = ast.parse(self.sample_python_code)
        structure = self.extractor._extract_code_structure(tree)
        
        self.assertIn('UserService', structure.classes)
        self.assertIn('create_user', structure.functions)
        self.assertGreater(structure.complexity_metrics['class_count'], 0)
    
    def test_design_pattern_detection(self):
        """Test design pattern detection."""
        import ast
        tree = ast.parse(self.sample_python_code)
        structure = self.extractor._extract_code_structure(tree)
        
        # Test repository pattern detection
        has_repository = self.extractor._detect_repository(tree, structure)
        self.assertTrue(has_repository)  # Should detect repository pattern
        
        # Test error handling pattern
        has_error_handling = self.extractor._detect_error_handling(tree, structure)
        self.assertTrue(has_error_handling)  # Should detect try/except
    
    def test_file_type_detection(self):
        """Test file type detection."""
        self.assertTrue(self.extractor._is_code_file('test.py'))
        self.assertTrue(self.extractor._is_code_file('test.js'))
        self.assertFalse(self.extractor._is_code_file('test.txt'))
        self.assertFalse(self.extractor._is_code_file('README.md'))


class TestWorkflowPatternExtractor(unittest.TestCase):
    """Test WorkflowPatternExtractor functionality."""
    
    def setUp(self):
        self.extractor = WorkflowPatternExtractor()
        
        self.completed_issue = {
            'issue_number': 33,
            'title': 'Implement caching system',
            'body': 'Add Redis-based caching for performance optimization',
            'complexity': 'high',
            'history': [
                {
                    'timestamp': '2023-01-01T10:00:00Z',
                    'label_added': {'name': 'state:new'}
                },
                {
                    'timestamp': '2023-01-01T11:00:00Z',
                    'label_added': {'name': 'state:analyzing'}
                },
                {
                    'timestamp': '2023-01-01T12:00:00Z',
                    'label_added': {'name': 'state:implementing'}
                },
                {
                    'timestamp': '2023-01-01T14:00:00Z',
                    'label_added': {'name': 'state:validating'}
                },
                {
                    'timestamp': '2023-01-01T15:00:00Z',
                    'label_added': {'name': 'state:complete'}
                }
            ],
            'agent_interactions': [
                {
                    'timestamp': '2023-01-01T11:00:00Z',
                    'agent': 'rif-analyst',
                    'type': 'analysis'
                },
                {
                    'timestamp': '2023-01-01T12:00:00Z',
                    'agent': 'rif-implementer',
                    'type': 'implementation'
                }
            ]
        }
    
    def test_initialization(self):
        """Test extractor initialization."""
        self.assertIsNotNone(self.extractor)
        self.assertIn('linear_workflow', self.extractor.workflow_patterns)
        self.assertIn('sequential_handoff', self.extractor.coordination_patterns)
        self.assertIn('checkpoint_pattern', self.extractor.quality_patterns)
    
    def test_workflow_sequence_extraction(self):
        """Test workflow sequence extraction from history."""
        workflow = self.extractor._extract_workflow_sequence(self.completed_issue)
        
        self.assertIsNotNone(workflow)
        self.assertIn('new', workflow.states)
        self.assertIn('complete', workflow.states)
        self.assertTrue(workflow.success)
        self.assertGreater(workflow.total_duration, 0)
    
    def test_state_extraction(self):
        """Test state extraction from label events."""
        event = {'label_added': {'name': 'state:implementing'}}
        state = self.extractor._extract_state_from_label_event(event)
        
        self.assertEqual(state, 'implementing')
    
    def test_workflow_pattern_detection(self):
        """Test workflow pattern detection."""
        workflow = self.extractor._extract_workflow_sequence(self.completed_issue)
        
        # Test cascade workflow detection (systematic progression)
        is_cascade = self.extractor._detect_cascade_workflow(workflow, self.completed_issue)
        self.assertTrue(is_cascade)  # Should detect systematic state progression
        
        # Test linear workflow detection
        is_linear = self.extractor._detect_linear_workflow(workflow, self.completed_issue)
        self.assertTrue(is_linear)  # No state repetition
    
    def test_coordination_pattern_detection(self):
        """Test coordination pattern detection."""
        workflow = self.extractor._extract_workflow_sequence(self.completed_issue)
        
        # Test sequential handoff
        is_sequential = self.extractor._detect_sequential_handoff(workflow, self.completed_issue)
        self.assertTrue(is_sequential)  # Different agents, no parallel activities
    
    def test_pattern_extraction(self):
        """Test full pattern extraction from completed issue."""
        patterns = self.extractor.extract_patterns(self.completed_issue)
        
        self.assertGreater(len(patterns), 0)
        
        # Should extract workflow patterns
        workflow_patterns = [p for p in patterns if p.pattern_type == 'workflow']
        self.assertGreater(len(workflow_patterns), 0)


class TestDecisionPatternExtractor(unittest.TestCase):
    """Test DecisionPatternExtractor functionality."""
    
    def setUp(self):
        self.extractor = DecisionPatternExtractor()
        
        self.completed_issue = {
            'issue_number': 44,
            'title': 'Database migration strategy',
            'body': 'We need to migrate from MySQL to PostgreSQL for better JSON support',
            'decisions': [
                {
                    'title': 'Choose PostgreSQL as target database',
                    'context': 'Current MySQL lacks advanced JSON features we need',
                    'decision': 'Migrate to PostgreSQL for better JSON and performance',
                    'alternatives': [
                        {
                            'name': 'Stay with MySQL',
                            'pros': ['No migration effort', 'Team familiarity'],
                            'cons': ['Limited JSON support', 'Performance issues']
                        },
                        {
                            'name': 'Move to PostgreSQL',
                            'pros': ['Better JSON support', 'Superior performance'],
                            'cons': ['Migration effort', 'Learning curve']
                        }
                    ],
                    'rationale': 'JSON features are critical for our use case',
                    'consequences': ['Improved performance', 'Migration overhead']
                }
            ],
            'comments': [
                {
                    'author': 'dev1',
                    'body': 'I think we should go with PostgreSQL because of the JSON support'
                },
                {
                    'author': 'dev2', 
                    'body': 'Agreed, the migration effort is worth it for the long-term benefits'
                }
            ]
        }
    
    def test_initialization(self):
        """Test extractor initialization."""
        self.assertIsNotNone(self.extractor)
        self.assertIn('architectural', self.extractor.decision_types)
        self.assertIn('trade_off_analysis', self.extractor.frameworks)
        self.assertIn('database', self.extractor.domain_keywords)
    
    def test_decision_record_parsing(self):
        """Test parsing of structured decision records."""
        decision_data = self.completed_issue['decisions'][0]
        pattern = self.extractor._parse_decision_record(decision_data, 'test_decision')
        
        self.assertIsNotNone(pattern)
        self.assertEqual(pattern.title, 'Choose PostgreSQL as target database')
        self.assertEqual(len(pattern.alternatives), 2)
        self.assertIn('MySQL', pattern.alternatives[0].name)
        self.assertIn('PostgreSQL', pattern.alternatives[1].name)
    
    def test_decision_type_classification(self):
        """Test decision type classification."""
        decision_data = self.completed_issue['decisions'][0]
        decision_type = self.extractor._classify_decision_type(decision_data)
        
        # Should classify as architectural or technical
        self.assertIn(decision_type, ['architectural', 'technical'])
    
    def test_framework_detection(self):
        """Test decision framework detection."""
        decision_data = self.completed_issue['decisions'][0]
        framework = self.extractor._identify_decision_framework(decision_data)
        
        # Should detect trade-off analysis (has alternatives with pros/cons)
        self.assertEqual(framework, 'trade_off_analysis')
    
    def test_domain_identification(self):
        """Test domain identification from text."""
        text = "database migration with JSON support and PostgreSQL"
        domain = self.extractor._identify_domain_from_text(text)
        
        self.assertEqual(domain, 'database')
    
    def test_pattern_extraction(self):
        """Test full pattern extraction from completed issue."""
        patterns = self.extractor.extract_patterns(self.completed_issue)
        
        self.assertGreater(len(patterns), 0)
        
        # Should extract decision patterns
        decision_patterns = [p for p in patterns if p.pattern_type == 'decision']
        self.assertGreater(len(decision_patterns), 0)
    
    def test_implicit_decision_extraction(self):
        """Test extraction of implicit decisions from text."""
        text = "We decided to use Redis for caching because of its performance benefits"
        decisions = self.extractor._extract_implicit_decisions_from_text(text)
        
        self.assertGreater(len(decisions), 0)
        self.assertIn('redis', decisions[0]['decision_text'].lower())


class TestSuccessMetricsCalculator(unittest.TestCase):
    """Test SuccessMetricsCalculator functionality."""
    
    def setUp(self):
        self.calculator = SuccessMetricsCalculator()
        
        # Create test pattern
        self.test_pattern = ExtractedPattern(
            title='Test Pattern',
            description='Pattern for testing metrics',
            pattern_type='test',
            source='test',
            content={},
            context={'complexity': 'medium', 'domain': 'web'},
            signature=PatternSignature.from_pattern({'title': 'Test Pattern'}),
            extraction_method='test',
            confidence=0.8,
            created_at=datetime.now()
        )
        
        # Create test application data
        self.application_data = [
            PatternApplication(
                pattern_id='test_pattern',
                application_id='app_1',
                context={'domain': 'web', 'project_size': 'medium'},
                success=True,
                performance_metrics={'execution_time': 0.5, 'memory_usage': 0.3},
                timestamp=datetime.now() - timedelta(days=30),
                feedback_score=0.8
            ),
            PatternApplication(
                pattern_id='test_pattern',
                application_id='app_2',
                context={'domain': 'api', 'project_size': 'large'},
                success=True,
                performance_metrics={'execution_time': 0.7, 'memory_usage': 0.4},
                timestamp=datetime.now() - timedelta(days=60),
                feedback_score=0.7
            ),
            PatternApplication(
                pattern_id='test_pattern',
                application_id='app_3',
                context={'domain': 'web', 'project_size': 'small'},
                success=False,
                performance_metrics={'execution_time': -0.2, 'memory_usage': 0.1},
                timestamp=datetime.now() - timedelta(days=90),
                feedback_score=0.3
            )
        ]
    
    def test_initialization(self):
        """Test calculator initialization."""
        self.assertIsNotNone(self.calculator)
        self.assertIn('high', self.calculator.confidence_levels)
        self.assertIn('success_rate', self.calculator.min_sample_sizes)
    
    def test_success_rate_calculation(self):
        """Test success rate calculation with confidence interval."""
        success_rate, confidence_interval = self.calculator._calculate_success_rate(self.application_data)
        
        self.assertEqual(success_rate, 2/3)  # 2 successes out of 3
        self.assertIsInstance(confidence_interval, tuple)
        self.assertEqual(len(confidence_interval), 2)
        self.assertLessEqual(confidence_interval[0], success_rate)
        self.assertGreaterEqual(confidence_interval[1], success_rate)
    
    def test_wilson_confidence_interval(self):
        """Test Wilson confidence interval calculation."""
        # Test with known values
        lower, upper = self.calculator._wilson_confidence_interval(8, 10, 0.95)
        
        self.assertGreaterEqual(lower, 0.0)
        self.assertLessEqual(upper, 1.0)
        self.assertLess(lower, upper)
    
    def test_applicability_score_calculation(self):
        """Test applicability score calculation."""
        score = self.calculator._calculate_applicability_score(self.test_pattern, self.application_data)
        
        self.assertGreaterEqual(score, 0.0)
        self.assertLessEqual(score, 1.0)
    
    def test_context_diversity_calculation(self):
        """Test context diversity calculation."""
        successful_contexts = [app.context for app in self.application_data if app.success]
        diversity = self.calculator._calculate_context_diversity(successful_contexts)
        
        self.assertGreaterEqual(diversity, 0.0)
        self.assertLessEqual(diversity, 1.0)
    
    def test_reusability_index_calculation(self):
        """Test reusability index calculation."""
        index = self.calculator._calculate_reusability_index(self.test_pattern, self.application_data)
        
        self.assertGreaterEqual(index, 0.0)
        self.assertLessEqual(index, 1.0)
    
    def test_reliability_score_calculation(self):
        """Test reliability score calculation."""
        score = self.calculator._calculate_reliability_score(self.application_data)
        
        self.assertGreaterEqual(score, 0.0)
        self.assertLessEqual(score, 1.0)
    
    def test_performance_impact_calculation(self):
        """Test performance impact calculation."""
        impact = self.calculator._calculate_performance_impact(self.application_data)
        
        self.assertGreaterEqual(impact, 0.0)
        self.assertLessEqual(impact, 1.0)
    
    def test_full_metrics_calculation(self):
        """Test full metrics calculation for a pattern."""
        metrics = self.calculator.calculate_pattern_metrics(self.test_pattern, self.application_data)
        
        self.assertIsInstance(metrics, SuccessMetrics)
        self.assertEqual(metrics.pattern_id, self.test_pattern.signature.combined_hash)
        self.assertEqual(metrics.sample_size, len(self.application_data))
        self.assertGreaterEqual(metrics.success_rate, 0.0)
        self.assertLessEqual(metrics.success_rate, 1.0)
    
    def test_pattern_ranking(self):
        """Test pattern ranking functionality."""
        patterns = [self.test_pattern]
        rankings = self.calculator.get_pattern_ranking(patterns)
        
        self.assertEqual(len(rankings), 1)
        self.assertEqual(rankings[0][0], self.test_pattern)
        self.assertIsInstance(rankings[0][1], float)
    
    def test_batch_metrics_calculation(self):
        """Test batch metrics calculation."""
        patterns = [self.test_pattern]
        metrics_batch = self.calculator.calculate_batch_metrics(patterns)
        
        self.assertEqual(len(metrics_batch), 1)
        self.assertIn(self.test_pattern.signature.combined_hash, metrics_batch)
    
    def test_metrics_serialization(self):
        """Test metrics serialization to/from dict."""
        metrics = self.calculator.calculate_pattern_metrics(self.test_pattern, self.application_data)
        
        # Test to_dict
        metrics_dict = metrics.to_dict()
        self.assertIsInstance(metrics_dict, dict)
        self.assertIn('success_rate', metrics_dict)
        self.assertIn('confidence_interval', metrics_dict)
        
        # Test from_dict
        restored_metrics = SuccessMetrics.from_dict(metrics_dict)
        self.assertEqual(restored_metrics.success_rate, metrics.success_rate)
        self.assertEqual(restored_metrics.pattern_id, metrics.pattern_id)


class TestIntegration(unittest.TestCase):
    """Integration tests for the complete pattern extraction system."""
    
    def setUp(self):
        """Set up integration test environment."""
        # Mock knowledge system
        self.mock_knowledge = Mock()
        self.mock_knowledge.store_knowledge.return_value = "pattern_123"
        self.mock_knowledge.retrieve_knowledge.return_value = [
            {'id': 'pattern_1', 'content': {'title': 'Test Pattern 1'}, 'metadata': {'type': 'extracted_pattern'}},
            {'id': 'pattern_2', 'content': {'title': 'Test Pattern 2'}, 'metadata': {'type': 'extracted_pattern'}}
        ]
        
        # Create main components
        self.discovery_engine = PatternDiscoveryEngine(self.mock_knowledge)
        self.code_extractor = CodePatternExtractor()
        self.workflow_extractor = WorkflowPatternExtractor()
        self.decision_extractor = DecisionPatternExtractor()
        self.metrics_calculator = SuccessMetricsCalculator()
        
        # Register extractors
        self.discovery_engine.register_extractor('code', self.code_extractor)
        self.discovery_engine.register_extractor('workflow', self.workflow_extractor)
        self.discovery_engine.register_extractor('decision', self.decision_extractor)
        
        # Comprehensive test issue
        self.comprehensive_issue = {
            'issue_number': 100,
            'title': 'Implement microservices architecture',
            'body': 'Refactor monolith into microservices for better scalability',
            'complexity': 'high',
            'code_changes': {
                'user_service.py': {
                    'added_lines': '''
class UserService:
    def __init__(self, repository, event_publisher):
        self.repository = repository
        self.event_publisher = event_publisher
    
    def create_user(self, user_data):
        try:
            user = self.repository.save(user_data)
            self.event_publisher.publish('user.created', user)
            return user
        except Exception as e:
            logger.error(f"Failed to create user: {e}")
            raise
'''
                }
            },
            'history': [
                {'timestamp': '2023-01-01T10:00:00Z', 'label_added': {'name': 'state:new'}},
                {'timestamp': '2023-01-01T11:00:00Z', 'label_added': {'name': 'state:analyzing'}},
                {'timestamp': '2023-01-01T12:00:00Z', 'label_added': {'name': 'state:architecting'}},
                {'timestamp': '2023-01-01T14:00:00Z', 'label_added': {'name': 'state:implementing'}},
                {'timestamp': '2023-01-01T16:00:00Z', 'label_added': {'name': 'state:validating'}},
                {'timestamp': '2023-01-01T17:00:00Z', 'label_added': {'name': 'state:complete'}}
            ],
            'decisions': [
                {
                    'title': 'Event-driven architecture',
                    'context': 'Need loose coupling between services',
                    'decision': 'Use event-driven architecture with message queues',
                    'alternatives': [
                        {'name': 'Direct API calls', 'pros': ['Simple'], 'cons': ['Tight coupling']},
                        {'name': 'Event-driven', 'pros': ['Loose coupling'], 'cons': ['Complexity']}
                    ],
                    'rationale': 'Better scalability and maintainability'
                }
            ],
            'comments': [
                {
                    'author': 'architect',
                    'body': 'We decided to implement event-driven architecture for better decoupling'
                }
            ],
            'agent_interactions': [
                {'timestamp': '2023-01-01T11:00:00Z', 'agent': 'rif-analyst'},
                {'timestamp': '2023-01-01T12:00:00Z', 'agent': 'rif-architect'},
                {'timestamp': '2023-01-01T14:00:00Z', 'agent': 'rif-implementer'}
            ]
        }
    
    def test_end_to_end_pattern_extraction(self):
        """Test complete end-to-end pattern extraction."""
        # Extract patterns using discovery engine
        patterns = self.discovery_engine.discover_patterns(self.comprehensive_issue)
        
        # Verify we got patterns from different extractors
        self.assertGreater(len(patterns), 0)
        
        pattern_types = {p.pattern_type for p in patterns}
        pattern_methods = {p.extraction_method for p in patterns}
        
        # Should have multiple pattern types
        self.assertGreaterEqual(len(pattern_types), 2)
        
        # Should have multiple extraction methods
        self.assertGreaterEqual(len(pattern_methods), 2)
        
        # Calculate metrics for extracted patterns
        for pattern in patterns:
            metrics = self.metrics_calculator.calculate_pattern_metrics(pattern)
            
            # Verify metrics are reasonable
            self.assertIsInstance(metrics, SuccessMetrics)
            self.assertGreaterEqual(metrics.success_rate, 0.0)
            self.assertLessEqual(metrics.success_rate, 1.0)
    
    def test_pattern_quality_assessment(self):
        """Test pattern quality assessment."""
        patterns = self.discovery_engine.discover_patterns(self.comprehensive_issue)
        
        if patterns:
            # Rank patterns by quality
            rankings = self.metrics_calculator.get_pattern_ranking(patterns)
            
            self.assertEqual(len(rankings), len(patterns))
            
            # Rankings should be in descending order
            scores = [score for _, score in rankings]
            self.assertEqual(scores, sorted(scores, reverse=True))
    
    def test_pattern_export(self):
        """Test pattern export functionality."""
        patterns = self.discovery_engine.discover_patterns(self.comprehensive_issue)
        
        if patterns:
            with tempfile.NamedTemporaryFile(mode='w', suffix='.json', delete=False) as tmp_file:
                tmp_path = tmp_file.name
            
            try:
                # Export patterns
                success = self.discovery_engine.export_patterns(tmp_path)
                self.assertTrue(success)
                
                # Verify export file exists and contains data
                self.assertTrue(os.path.exists(tmp_path))
                
                with open(tmp_path, 'r') as f:
                    export_data = json.load(f)
                
                self.assertIn('patterns', export_data)
                self.assertGreater(export_data['pattern_count'], 0)
                
            finally:
                if os.path.exists(tmp_path):
                    os.unlink(tmp_path)
    
    def test_metrics_report_export(self):
        """Test metrics report export."""
        patterns = self.discovery_engine.discover_patterns(self.comprehensive_issue)
        
        if patterns:
            with tempfile.NamedTemporaryFile(mode='w', suffix='.json', delete=False) as tmp_file:
                tmp_path = tmp_file.name
            
            try:
                # Export metrics report
                success = self.metrics_calculator.export_metrics_report(patterns, tmp_path)
                self.assertTrue(success)
                
                # Verify report file exists and contains data
                self.assertTrue(os.path.exists(tmp_path))
                
                with open(tmp_path, 'r') as f:
                    report_data = json.load(f)
                
                self.assertIn('report_metadata', report_data)
                self.assertIn('summary_statistics', report_data)
                self.assertIn('pattern_rankings', report_data)
                
            finally:
                if os.path.exists(tmp_path):
                    os.unlink(tmp_path)
    
    def test_error_handling(self):
        """Test error handling in the extraction system."""
        # Test with malformed issue data
        malformed_issue = {'issue_number': None}
        
        # Should not crash and should return empty patterns
        patterns = self.discovery_engine.discover_patterns(malformed_issue)
        self.assertEqual(len(patterns), 0)
        
        # Should track errors in statistics
        self.assertGreater(self.discovery_engine.extraction_stats['errors_encountered'], 0)
    
    def test_performance_benchmarks(self):
        """Test performance benchmarks for the system."""
        import time
        
        # Measure extraction time
        start_time = time.time()
        patterns = self.discovery_engine.discover_patterns(self.comprehensive_issue)
        extraction_time = time.time() - start_time
        
        # Should complete within reasonable time (2 seconds for this test)
        self.assertLess(extraction_time, 2.0)
        
        # Measure metrics calculation time
        if patterns:
            start_time = time.time()
            self.metrics_calculator.calculate_batch_metrics(patterns)
            metrics_time = time.time() - start_time
            
            # Should complete within reasonable time
            self.assertLess(metrics_time, 1.0)


if __name__ == '__main__':
    # Configure logging for tests
    logging.basicConfig(level=logging.WARNING)
    
    # Run tests with verbose output
    unittest.main(verbosity=2)