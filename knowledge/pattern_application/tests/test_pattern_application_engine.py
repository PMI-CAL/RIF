"""
Tests for Pattern Application Engine

Comprehensive tests for the main PatternApplicationEngine class,
including pattern application workflows, error handling, and integration scenarios.
"""

import pytest
import json
from datetime import datetime
from unittest.mock import Mock, MagicMock, patch

from knowledge.pattern_application.core import (
    Pattern, IssueContext, TechStack, IssueConstraints,
    ApplicationRecord, PatternApplicationStatus,
    PatternApplicationError, PatternNotFoundError
)
from knowledge.pattern_application.engine import PatternApplicationEngine
from knowledge.pattern_application.context_extractor import ContextExtractor
from knowledge.pattern_application.pattern_matcher import BasicPatternMatcher


class TestPatternApplicationEngine:
    """Test suite for PatternApplicationEngine."""
    
    @pytest.fixture
    def mock_knowledge_system(self):
        """Mock knowledge system for testing."""
        mock_system = Mock()
        mock_system.retrieve_knowledge.return_value = [
            {
                'id': 'pattern_123',
                'content': {
                    'pattern_id': 'pattern_123',
                    'pattern_name': 'Test Pattern',
                    'description': 'A test pattern for unit testing',
                    'complexity': 'medium',
                    'tech_stack': {
                        'primary_language': 'python',
                        'frameworks': ['fastapi'],
                        'databases': ['postgresql'],
                        'tools': ['docker'],
                        'architecture_pattern': 'rest',
                        'deployment_target': 'cloud'
                    },
                    'domain': 'backend',
                    'tags': ['api', 'database'],
                    'confidence': 0.85,
                    'success_rate': 0.78,
                    'usage_count': 12,
                    'implementation_steps': [
                        'Setup development environment',
                        'Create API endpoints',
                        'Implement database models',
                        'Add tests'
                    ],
                    'validation_criteria': ['All tests pass', 'API endpoints working']
                }
            }
        ]
        mock_system.store_knowledge.return_value = 'stored_id'
        return mock_system
    
    @pytest.fixture
    def mock_pattern_matcher(self):
        """Mock pattern matcher for testing."""
        matcher = Mock(spec=BasicPatternMatcher)
        
        # Create a test pattern
        test_pattern = Pattern(
            pattern_id='pattern_123',
            name='Test Pattern',
            description='A test pattern for unit testing',
            complexity='medium',
            tech_stack=TechStack(
                primary_language='python',
                frameworks=['fastapi'],
                databases=['postgresql'],
                tools=['docker'],
                architecture_pattern='rest',
                deployment_target='cloud'
            ),
            domain='backend',
            tags=['api', 'database'],
            confidence=0.85,
            success_rate=0.78,
            usage_count=12,
            implementation_steps=[
                {'description': 'Setup development environment'},
                {'description': 'Create API endpoints'},
                {'description': 'Implement database models'},
                {'description': 'Add tests'}
            ],
            validation_criteria=['All tests pass', 'API endpoints working']
        )
        
        matcher.find_applicable_patterns.return_value = [test_pattern]
        matcher.rank_patterns.return_value = [test_pattern]
        matcher.calculate_pattern_relevance.return_value = 0.85
        
        return matcher
    
    @pytest.fixture
    def mock_context_extractor(self):
        """Mock context extractor for testing."""
        extractor = Mock(spec=ContextExtractor)
        
        test_context = IssueContext(
            issue_id='123',
            title='Test Issue',
            description='A test issue for unit testing',
            complexity='medium',
            tech_stack=TechStack(
                primary_language='python',
                frameworks=['fastapi'],
                databases=['postgresql']
            ),
            constraints=IssueConstraints(),
            domain='backend',
            labels=['api', 'enhancement']
        )
        
        extractor.extract_issue_context.return_value = test_context
        
        return extractor
    
    @pytest.fixture
    def engine(self, mock_knowledge_system, mock_pattern_matcher, mock_context_extractor):
        """Create PatternApplicationEngine with mocked dependencies."""
        return PatternApplicationEngine(
            pattern_matcher=mock_pattern_matcher,
            context_extractor=mock_context_extractor,
            knowledge_system=mock_knowledge_system
        )
    
    def test_engine_initialization(self, mock_pattern_matcher, mock_context_extractor, mock_knowledge_system):
        """Test engine initialization."""
        engine = PatternApplicationEngine(
            pattern_matcher=mock_pattern_matcher,
            context_extractor=mock_context_extractor,
            knowledge_system=mock_knowledge_system
        )
        
        assert engine.pattern_matcher == mock_pattern_matcher
        assert engine.context_extractor == mock_context_extractor
        assert engine.knowledge_system == mock_knowledge_system
        assert isinstance(engine.applications, dict)
        assert 'max_patterns_to_consider' in engine.config
    
    def test_apply_pattern_success(self, engine):
        """Test successful pattern application."""
        # Create test context
        context = IssueContext(
            issue_id='123',
            title='Test Issue',
            description='Test implementation',
            complexity='medium',
            tech_stack=TechStack(primary_language='python'),
            constraints=IssueConstraints(),
            domain='backend'
        )
        
        # Apply pattern
        result = engine.apply_pattern('pattern_123', context)
        
        # Verify result
        assert isinstance(result, ApplicationRecord)
        assert result.pattern_id == 'pattern_123'
        assert result.issue_id == '123'
        assert result.status == PatternApplicationStatus.COMPLETED
        assert result.adaptation_result is not None
        assert result.implementation_plan is not None
        assert result.application_id in engine.applications
    
    def test_apply_pattern_not_found(self, engine, mock_knowledge_system):
        """Test pattern application with non-existent pattern."""
        # Mock empty result
        mock_knowledge_system.retrieve_knowledge.return_value = []
        
        context = IssueContext(
            issue_id='123',
            title='Test Issue',
            description='Test implementation',
            complexity='medium',
            tech_stack=TechStack(primary_language='python'),
            constraints=IssueConstraints(),
            domain='backend'
        )
        
        # Expect PatternNotFoundError
        with pytest.raises(PatternApplicationError):
            engine.apply_pattern('nonexistent_pattern', context)
    
    def test_apply_best_pattern_success(self, engine):
        """Test finding and applying best pattern."""
        context = IssueContext(
            issue_id='123',
            title='Test Issue',
            description='Test implementation',
            complexity='medium',
            tech_stack=TechStack(primary_language='python'),
            constraints=IssueConstraints(),
            domain='backend'
        )
        
        result = engine.apply_best_pattern(context)
        
        assert result is not None
        assert isinstance(result, ApplicationRecord)
        assert result.status == PatternApplicationStatus.COMPLETED
    
    def test_apply_best_pattern_no_patterns(self, engine, mock_pattern_matcher):
        """Test best pattern application when no patterns found."""
        # Mock no patterns found
        mock_pattern_matcher.find_applicable_patterns.return_value = []
        
        context = IssueContext(
            issue_id='123',
            title='Test Issue',
            description='Test implementation',
            complexity='medium',
            tech_stack=TechStack(primary_language='python'),
            constraints=IssueConstraints(),
            domain='backend'
        )
        
        result = engine.apply_best_pattern(context)
        
        assert result is None
    
    def test_apply_best_pattern_low_confidence(self, engine, mock_pattern_matcher):
        """Test best pattern application with low confidence pattern."""
        # Mock low relevance score
        mock_pattern_matcher.calculate_pattern_relevance.return_value = 0.3
        
        context = IssueContext(
            issue_id='123',
            title='Test Issue',
            description='Test implementation',
            complexity='medium',
            tech_stack=TechStack(primary_language='python'),
            constraints=IssueConstraints(),
            domain='backend'
        )
        
        result = engine.apply_best_pattern(context)
        
        assert result is None  # Should reject due to low confidence
    
    def test_adapt_pattern_to_context(self, engine):
        """Test pattern adaptation to context."""
        # Create test pattern and context
        pattern = Pattern(
            pattern_id='test_pattern',
            name='Test Pattern',
            description='Test pattern',
            complexity='low',
            tech_stack=TechStack(primary_language='javascript'),
            domain='frontend'
        )
        
        context = IssueContext(
            issue_id='123',
            title='Test Issue',
            description='Test implementation',
            complexity='high',
            tech_stack=TechStack(primary_language='python'),
            constraints=IssueConstraints(),
            domain='backend'
        )
        
        result = engine.adapt_pattern_to_context(pattern, context)
        
        assert result is not None
        assert result.adapted_pattern is not None
        assert len(result.changes_made) > 0
        assert result.confidence_score > 0
        assert result.adaptation_notes is not None
    
    def test_track_application_progress(self, engine):
        """Test application progress tracking."""
        # Create application first
        context = IssueContext(
            issue_id='123',
            title='Test Issue',
            description='Test implementation',
            complexity='medium',
            tech_stack=TechStack(primary_language='python'),
            constraints=IssueConstraints(),
            domain='backend'
        )
        
        application = engine.apply_pattern('pattern_123', context)
        
        # Track progress
        success = engine.track_application_progress(
            application.application_id,
            PatternApplicationStatus.IN_PROGRESS,
            {'tasks_completed': 2, 'quality_score': 0.8}
        )
        
        assert success is True
        assert application.status == PatternApplicationStatus.IN_PROGRESS
        assert 'tasks_completed' in application.execution_metrics
        assert 'quality_score' in application.execution_metrics
    
    def test_track_application_progress_not_found(self, engine):
        """Test tracking progress for non-existent application."""
        success = engine.track_application_progress(
            'nonexistent_id',
            PatternApplicationStatus.IN_PROGRESS
        )
        
        assert success is False
    
    def test_measure_application_success(self, engine):
        """Test application success measurement."""
        # Create completed application
        context = IssueContext(
            issue_id='123',
            title='Test Issue',
            description='Test implementation',
            complexity='medium',
            tech_stack=TechStack(primary_language='python'),
            constraints=IssueConstraints(),
            domain='backend'
        )
        
        application = engine.apply_pattern('pattern_123', context)
        
        # Measure success
        success_score = engine.measure_application_success(application.application_id)
        
        assert isinstance(success_score, float)
        assert 0.0 <= success_score <= 1.0
        assert application.success_score is not None
    
    def test_measure_application_success_not_found(self, engine):
        """Test success measurement for non-existent application."""
        success_score = engine.measure_application_success('nonexistent_id')
        
        assert success_score == 0.0
    
    def test_get_application_record(self, engine):
        """Test retrieving application record."""
        # Create application
        context = IssueContext(
            issue_id='123',
            title='Test Issue',
            description='Test implementation',
            complexity='medium',
            tech_stack=TechStack(primary_language='python'),
            constraints=IssueConstraints(),
            domain='backend'
        )
        
        application = engine.apply_pattern('pattern_123', context)
        
        # Retrieve record
        retrieved = engine.get_application_record(application.application_id)
        
        assert retrieved is not None
        assert retrieved.application_id == application.application_id
    
    def test_get_application_record_not_found(self, engine):
        """Test retrieving non-existent application record."""
        retrieved = engine.get_application_record('nonexistent_id')
        
        assert retrieved is None
    
    def test_list_applications(self, engine):
        """Test listing all applications."""
        # Create multiple applications
        context1 = IssueContext(
            issue_id='123',
            title='Test Issue 1',
            description='Test implementation 1',
            complexity='medium',
            tech_stack=TechStack(primary_language='python'),
            constraints=IssueConstraints(),
            domain='backend'
        )
        
        context2 = IssueContext(
            issue_id='456',
            title='Test Issue 2',
            description='Test implementation 2',
            complexity='high',
            tech_stack=TechStack(primary_language='javascript'),
            constraints=IssueConstraints(),
            domain='frontend'
        )
        
        app1 = engine.apply_pattern('pattern_123', context1)
        app2 = engine.apply_pattern('pattern_123', context2)
        
        # List all applications
        all_apps = engine.list_applications()
        
        assert len(all_apps) == 2
        assert app1 in all_apps
        assert app2 in all_apps
    
    def test_list_applications_filtered(self, engine):
        """Test listing applications filtered by issue ID."""
        # Create multiple applications
        context1 = IssueContext(
            issue_id='123',
            title='Test Issue 1',
            description='Test implementation 1',
            complexity='medium',
            tech_stack=TechStack(primary_language='python'),
            constraints=IssueConstraints(),
            domain='backend'
        )
        
        context2 = IssueContext(
            issue_id='456',
            title='Test Issue 2',
            description='Test implementation 2',
            complexity='high',
            tech_stack=TechStack(primary_language='javascript'),
            constraints=IssueConstraints(),
            domain='frontend'
        )
        
        app1 = engine.apply_pattern('pattern_123', context1)
        app2 = engine.apply_pattern('pattern_123', context2)
        
        # List applications for specific issue
        filtered_apps = engine.list_applications(issue_id='123')
        
        assert len(filtered_apps) == 1
        assert app1 in filtered_apps
        assert app2 not in filtered_apps
    
    def test_configuration_loading(self, engine):
        """Test that configuration is loaded correctly."""
        config = engine.config
        
        assert 'max_patterns_to_consider' in config
        assert 'min_confidence_threshold' in config
        assert 'adaptation_strategies' in config
        assert 'success_tracking_enabled' in config
        assert 'store_applications' in config
        
        # Check default values
        assert isinstance(config['max_patterns_to_consider'], int)
        assert isinstance(config['min_confidence_threshold'], float)
        assert isinstance(config['adaptation_strategies'], list)
        assert isinstance(config['success_tracking_enabled'], bool)
        assert isinstance(config['store_applications'], bool)


if __name__ == '__main__':
    pytest.main([__file__, '-v'])