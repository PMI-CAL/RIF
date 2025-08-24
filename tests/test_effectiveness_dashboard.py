#!/usr/bin/env python3
"""
Test suite for Effectiveness Dashboard
Issue #94: Quality Gate Effectiveness Monitoring
"""

import os
import sys
import json
import tempfile
import pytest
from datetime import datetime, timedelta
from pathlib import Path

# Add the claude/commands directory to the Python path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'claude', 'commands'))

from effectiveness_dashboard import (
    EffectivenessDashboard,
    DashboardMetrics,
    TrendData,
    ComponentBreakdown,
    DashboardData
)

class TestEffectivenessDashboard:
    """Test suite for EffectivenessDashboard class."""
    
    def setup_method(self):
        """Setup test environment before each test."""
        self.temp_dir = tempfile.mkdtemp()
        self.dashboard = EffectivenessDashboard(knowledge_base_path=self.temp_dir)
        
        # Create mock metrics data
        self._create_mock_metrics_data()
    
    def teardown_method(self):
        """Cleanup after each test."""
        import shutil
        shutil.rmtree(self.temp_dir, ignore_errors=True)
    
    def _create_mock_metrics_data(self):
        """Create mock metrics data for testing."""
        # Create quality_metrics directory structure
        metrics_dir = self.dashboard.metrics_path
        recent_dir = metrics_dir / "recent"
        recent_dir.mkdir(parents=True, exist_ok=True)
        
        # Create mock decision data
        mock_decisions = []
        
        # Generate 30 days of mock data
        base_date = datetime.now() - timedelta(days=30)
        
        for day in range(30):
            current_date = base_date + timedelta(days=day)
            
            # Generate 5-15 decisions per day
            num_decisions = 5 + (day % 10)  # Varies between 5-15
            
            for decision_idx in range(num_decisions):
                # Create realistic decision data
                decision = {
                    'timestamp': (current_date + timedelta(hours=decision_idx)).isoformat(),
                    'issue_number': 1000 + (day * 100) + decision_idx,
                    'gate_type': ['code_coverage', 'security_scan', 'performance', 'linting'][decision_idx % 4],
                    'decision': self._get_mock_decision(day, decision_idx),
                    'outcome': self._get_mock_outcome(day, decision_idx),
                    'component_type': ['business_logic', 'ui_components', 'public_apis', 'integration_code'][decision_idx % 4],
                    'processing_time_ms': 20 + (decision_idx * 5) + (day % 10),
                    'confidence_score': 0.8 + (0.1 * (decision_idx % 3)),
                    'manual_intervention': decision_idx % 8 == 0,  # 1 in 8 decisions
                    'intervention_appropriate': decision_idx % 10 != 9,  # Most interventions appropriate
                    'error': decision_idx % 20 == 19  # 5% error rate
                }
                
                mock_decisions.append(decision)
        
        # Save mock data to file - individual decision records
        for i, decision in enumerate(mock_decisions):
            mock_file = recent_dir / f"mock_metrics_{i}.json"
            with open(mock_file, 'w') as f:
                json.dump(decision, f)
        
        # Also create a session file in realtime directory
        realtime_dir = metrics_dir / "realtime"
        realtime_dir.mkdir(parents=True, exist_ok=True)
        
        session_file = realtime_dir / "session_test.json"
        with open(session_file, 'w') as f:
            session_data = {
                'session_id': 'test_session',
                'timestamp': datetime.now().isoformat(),
                'decisions': mock_decisions
            }
            json.dump(session_data, f)
    
    def _get_mock_decision(self, day: int, decision_idx: int) -> str:
        """Generate mock decision based on day and index."""
        # Create patterns to simulate realistic decision distribution
        if day < 10:  # Earlier days have more failures
            return ['pass', 'fail', 'pass', 'concerns'][decision_idx % 4]
        elif day < 20:  # Middle period improves
            return ['pass', 'pass', 'concerns', 'pass'][decision_idx % 4]
        else:  # Recent period is better
            return ['pass', 'pass', 'pass', 'concerns'][decision_idx % 4]
    
    def _get_mock_outcome(self, day: int, decision_idx: int) -> str:
        """Generate mock outcome based on decision quality."""
        # Correlate outcomes with decisions for realistic metrics
        decision = self._get_mock_decision(day, decision_idx)
        
        if decision == 'pass':
            # Most passes are true negatives, some false negatives
            return 'true_negative' if decision_idx % 8 != 0 else 'false_negative'
        elif decision == 'fail':
            # Most failures are true positives, some false positives
            return 'true_positive' if decision_idx % 6 != 0 else 'false_positive'
        elif decision == 'concerns':
            # Mixed outcomes for concerns
            return ['true_negative', 'true_positive'][decision_idx % 2]
        else:
            return 'true_negative'
    
    def test_dashboard_initialization(self):
        """Test dashboard initializes correctly."""
        assert self.dashboard.knowledge_path.exists()
        assert self.dashboard.metrics_path.exists()
        assert self.dashboard.dashboard_path.exists()
    
    def test_generate_dashboard_data(self):
        """Test generating complete dashboard data."""
        dashboard_data = self.dashboard.generate_dashboard_data(days=30)
        
        # Check data structure
        assert isinstance(dashboard_data, DashboardData)
        assert isinstance(dashboard_data.summary_metrics, DashboardMetrics)
        assert isinstance(dashboard_data.trend_data, TrendData)
        assert isinstance(dashboard_data.component_breakdown, list)
        assert isinstance(dashboard_data.recent_insights, list)
        assert isinstance(dashboard_data.alerts, list)
        
        # Check summary metrics
        metrics = dashboard_data.summary_metrics
        assert 0 <= metrics.overall_effectiveness <= 1
        assert 0 <= metrics.false_positive_rate <= 1
        assert 0 <= metrics.false_negative_rate <= 1
        assert 0 <= metrics.intervention_accuracy <= 1
        assert metrics.total_decisions > 0
        assert metrics.avg_response_time >= 0
        assert metrics.trend_direction in ['improving', 'stable', 'declining', 'insufficient_data']
        
        # Check trend data
        trends = dashboard_data.trend_data
        assert len(trends.timestamps) == 30  # 30 days
        assert len(trends.effectiveness_scores) == 30
        assert len(trends.false_positive_rates) == 30
        assert len(trends.false_negative_rates) == 30
        assert len(trends.decision_volumes) == 30
        
        # Check component breakdown
        assert len(dashboard_data.component_breakdown) > 0
        for component in dashboard_data.component_breakdown:
            assert isinstance(component, ComponentBreakdown)
            assert component.component_name in ['business_logic', 'ui_components', 'public_apis', 'integration_code']
            assert 0 <= component.effectiveness_score <= 1
            assert component.decision_count > 0
            assert component.avg_processing_time >= 0
            assert 0 <= component.success_rate <= 1
            assert component.trend in ['improving', 'declining', 'stable', 'insufficient_data']
    
    def test_dashboard_with_no_data(self):
        """Test dashboard handles empty data gracefully."""
        # Create dashboard with empty metrics
        empty_dashboard = EffectivenessDashboard(knowledge_base_path=tempfile.mkdtemp())
        dashboard_data = empty_dashboard.generate_dashboard_data(days=7)
        
        # Should return empty dashboard structure
        assert isinstance(dashboard_data, DashboardData)
        assert dashboard_data.summary_metrics.total_decisions == 0
        assert dashboard_data.summary_metrics.overall_effectiveness == 0.0
        assert len(dashboard_data.trend_data.timestamps) == 0
        assert len(dashboard_data.component_breakdown) == 0
        assert "No recent quality gate data available" in ' '.join(dashboard_data.recent_insights)
    
    def test_trend_calculation(self):
        """Test trend direction calculation."""
        dashboard_data = self.dashboard.generate_dashboard_data(days=30)
        
        # With our mock data, we expect improving trend (failures decrease over time)
        assert dashboard_data.summary_metrics.trend_direction in ['improving', 'stable']
        
        # Check trend data has proper time series
        trends = dashboard_data.trend_data
        assert all(isinstance(score, (int, float)) for score in trends.effectiveness_scores)
        assert all(isinstance(rate, (int, float)) for rate in trends.false_positive_rates)
        assert all(isinstance(rate, (int, float)) for rate in trends.false_negative_rates)
        assert all(isinstance(vol, int) for vol in trends.decision_volumes)
    
    def test_component_analysis(self):
        """Test component performance breakdown."""
        dashboard_data = self.dashboard.generate_dashboard_data(days=30)
        
        # Should have data for all component types in mock data
        components = {comp.component_name for comp in dashboard_data.component_breakdown}
        expected_components = {'business_logic', 'ui_components', 'public_apis', 'integration_code'}
        assert components == expected_components
        
        # Components should be sorted by decision count
        decision_counts = [comp.decision_count for comp in dashboard_data.component_breakdown]
        assert decision_counts == sorted(decision_counts, reverse=True)
        
        # Each component should have reasonable metrics
        for component in dashboard_data.component_breakdown:
            assert 0 <= component.effectiveness_score <= 1
            assert component.decision_count > 0
            assert component.avg_processing_time > 0
            assert 0 <= component.success_rate <= 1
    
    def test_insights_generation(self):
        """Test generation of actionable insights."""
        dashboard_data = self.dashboard.generate_dashboard_data(days=30)
        
        # Should generate meaningful insights
        assert len(dashboard_data.recent_insights) > 0
        assert len(dashboard_data.recent_insights) <= 5  # Limited to top 5
        
        # Insights should be strings
        assert all(isinstance(insight, str) for insight in dashboard_data.recent_insights)
        assert all(len(insight) > 10 for insight in dashboard_data.recent_insights)  # Non-trivial insights
    
    def test_alerts_checking(self):
        """Test alert generation for concerning metrics."""
        dashboard_data = self.dashboard.generate_dashboard_data(days=30)
        
        # Alerts should be list of strings
        assert isinstance(dashboard_data.alerts, list)
        assert all(isinstance(alert, str) for alert in dashboard_data.alerts)
        
        # Check alert logic for high false positive rate
        if dashboard_data.summary_metrics.false_positive_rate > 0.2:
            alert_found = any("false positive rate" in alert.lower() for alert in dashboard_data.alerts)
            assert alert_found
        
        # Check alert logic for high false negative rate
        if dashboard_data.summary_metrics.false_negative_rate > 0.05:
            alert_found = any("false negative rate" in alert.lower() for alert in dashboard_data.alerts)
            assert alert_found
    
    def test_html_report_generation(self):
        """Test HTML report generation."""
        dashboard_data = self.dashboard.generate_dashboard_data(days=7)
        html_report = self.dashboard.generate_html_report(dashboard_data)
        
        # Should generate valid HTML
        assert html_report.startswith('<!DOCTYPE html>')
        assert '</html>' in html_report
        assert 'Quality Gate Effectiveness Dashboard' in html_report
        
        # Should include key metrics
        assert str(dashboard_data.summary_metrics.overall_effectiveness) in html_report
        assert str(dashboard_data.summary_metrics.total_decisions) in html_report
        assert str(int(dashboard_data.summary_metrics.avg_response_time)) in html_report
        
        # Should include component breakdown
        for component in dashboard_data.component_breakdown:
            assert component.component_name in html_report
        
        # Should include insights
        for insight in dashboard_data.recent_insights:
            assert insight in html_report
    
    def test_dashboard_data_persistence(self):
        """Test that dashboard data is saved correctly."""
        dashboard_data = self.dashboard.generate_dashboard_data(days=7)
        
        # Check current dashboard file exists
        current_file = self.dashboard.dashboard_path / "current_dashboard.json"
        assert current_file.exists()
        
        # Check historical file exists
        historical_files = list(self.dashboard.dashboard_path.glob("dashboard_*.json"))
        assert len(historical_files) >= 1
        
        # Check file contains correct data
        with open(current_file, 'r') as f:
            saved_data = json.load(f)
        
        assert saved_data['summary_metrics']['total_decisions'] == dashboard_data.summary_metrics.total_decisions
        assert len(saved_data['component_breakdown']) == len(dashboard_data.component_breakdown)
        assert len(saved_data['recent_insights']) == len(dashboard_data.recent_insights)
    
    def test_performance_metrics(self):
        """Test performance tracking of dashboard generation."""
        start_time = datetime.now()
        dashboard_data = self.dashboard.generate_dashboard_data(days=30)
        end_time = datetime.now()
        
        # Dashboard generation should be reasonably fast
        generation_time = (end_time - start_time).total_seconds()
        assert generation_time < 5.0  # Should complete in under 5 seconds
        
        # Check processing metrics are reasonable
        assert dashboard_data.summary_metrics.avg_response_time < 1000  # Under 1 second per decision
    
    def test_edge_cases(self):
        """Test various edge cases."""
        # Test with 0 days
        dashboard_data = self.dashboard.generate_dashboard_data(days=0)
        assert dashboard_data.summary_metrics.total_decisions >= 0
        
        # Test with very large days value
        dashboard_data = self.dashboard.generate_dashboard_data(days=365)
        assert isinstance(dashboard_data, DashboardData)
        
        # Test with corrupted metrics file (should handle gracefully)
        corrupted_file = self.dashboard.metrics_path / "recent" / "corrupted.json"
        corrupted_file.parent.mkdir(parents=True, exist_ok=True)
        with open(corrupted_file, 'w') as f:
            f.write("invalid json content")
        
        # Should still generate dashboard without crashing
        dashboard_data = self.dashboard.generate_dashboard_data(days=7)
        assert isinstance(dashboard_data, DashboardData)


if __name__ == '__main__':
    pytest.main([__file__, '-v'])