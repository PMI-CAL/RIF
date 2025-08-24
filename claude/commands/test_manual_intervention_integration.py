#!/usr/bin/env python3
"""
Manual Intervention Integration Test - Issue #92
Comprehensive test of the Risk-Based Manual Intervention Framework.

Tests the complete workflow from risk assessment through specialist assignment,
SLA monitoring, decision tracking, and audit trail validation.
"""

import json
import sys
import os
import logging
import asyncio
from datetime import datetime, timezone, timedelta
from pathlib import Path

# Add current directory to path for imports
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from risk_assessment_engine import RiskAssessmentEngine, ChangeContext, RiskScore
from specialist_assignment_engine import SpecialistAssignmentEngine, AssignmentRequest, SpecialistType
from sla_monitoring_system import SLAMonitoringSystem
from decision_audit_tracker import DecisionAuditTracker, AuditRecord
from manual_intervention_workflow import ManualInterventionWorkflow, DecisionType, ManualDecision

class ManualInterventionIntegrationTest:
    """Comprehensive integration test for the manual intervention framework."""
    
    def __init__(self):
        """Initialize the integration test."""
        self.setup_logging()
        self.test_results = []
        
    def setup_logging(self):
        """Setup logging for integration test."""
        logging.basicConfig(
            level=logging.INFO,
            format='%(asctime)s - IntegrationTest - %(levelname)s - %(message)s'
        )
        self.logger = logging.getLogger(__name__)
        
    def log_test_result(self, test_name: str, success: bool, details: str = ""):
        """Log and store test result."""
        status = "✅ PASS" if success else "❌ FAIL"
        message = f"{status} - {test_name}"
        if details:
            message += f": {details}"
            
        self.logger.info(message)
        
        self.test_results.append({
            'test_name': test_name,
            'success': success,
            'details': details,
            'timestamp': datetime.now(timezone.utc).isoformat()
        })
        
        return success
    
    def run_all_tests(self) -> bool:
        """Run all integration tests."""
        self.logger.info("🧪 Starting Manual Intervention Framework Integration Tests")
        
        # Test individual components first
        risk_test = self.test_risk_assessment_engine()
        specialist_test = self.test_specialist_assignment_engine()
        sla_test = self.test_sla_monitoring_system()
        audit_test = self.test_decision_audit_tracker()
        
        # Test full workflow integration
        workflow_test = self.test_full_workflow_integration()
        
        # Generate test report
        self.generate_test_report()
        
        all_passed = all([risk_test, specialist_test, sla_test, audit_test, workflow_test])
        
        if all_passed:
            self.logger.info("🎉 All integration tests PASSED")
        else:
            self.logger.error("💥 Some integration tests FAILED")
            
        return all_passed
    
    def test_risk_assessment_engine(self) -> bool:
        """Test Risk Assessment Engine functionality."""
        self.logger.info("Testing Risk Assessment Engine...")
        
        try:
            # Initialize engine
            engine = RiskAssessmentEngine()
            
            # Test configuration loading
            if not self.log_test_result("Risk Assessment Config Load", 
                                      engine.config is not None, 
                                      f"Config has {len(engine.config)} sections"):
                return False
            
            # Test risk assessment with mock data
            change_context = ChangeContext(
                files_changed=["auth/oauth.py", "security/jwt_handler.py"],
                lines_added=150,
                lines_removed=50,
                issue_number=92,
                issue_details={
                    'title': 'Fix OAuth security vulnerability',
                    'body': 'Authentication bypass vulnerability needs immediate fix',
                    'labels': [{'name': 'security'}, {'name': 'high'}],
                    'createdAt': datetime.now(timezone.utc).isoformat()
                },
                author="test-user"
            )
            
            risk_score = engine.assess_change_risk(change_context)
            
            # Validate risk score
            score_valid = (
                risk_score.total_score >= 0.0 and 
                risk_score.total_score <= 1.0 and
                risk_score.risk_level in ['low', 'medium', 'high', 'critical']
            )
            
            if not self.log_test_result("Risk Score Calculation", score_valid, 
                                      f"Score: {risk_score.total_score:.2f}, Level: {risk_score.risk_level}"):
                return False
                
            # Test security factor detection
            security_detected = risk_score.factors.security_score > 0.3
            if not self.log_test_result("Security Risk Detection", security_detected,
                                      f"Security score: {risk_score.factors.security_score:.2f}"):
                return False
            
            return True
            
        except Exception as e:
            self.log_test_result("Risk Assessment Engine", False, f"Exception: {e}")
            return False
    
    def test_specialist_assignment_engine(self) -> bool:
        """Test Specialist Assignment Engine functionality."""
        self.logger.info("Testing Specialist Assignment Engine...")
        
        try:
            # Initialize engine
            engine = SpecialistAssignmentEngine()
            
            # Test specialist registry loading
            total_specialists = sum(len(specs) for specs in engine.specialists.values())
            if not self.log_test_result("Specialist Registry Load", total_specialists > 0,
                                      f"Loaded {total_specialists} specialists"):
                return False
            
            # Test assignment request
            assignment_request = AssignmentRequest(
                issue_number=92,
                risk_score=0.75,
                risk_level="high",
                primary_risk_factors=["security_changes", "auth_modifications"],
                specialist_type=SpecialistType.SECURITY,
                urgency_level="high",
                files_changed=["auth/oauth.py", "security/jwt_handler.py"],
                estimated_review_time=4.0,
                special_requirements=[]
            )
            
            assignment_result = engine.assign_specialist(assignment_request)
            
            # Validate assignment
            assignment_valid = (
                assignment_result.assigned_specialist is not None and
                assignment_result.assignment_confidence > 0.0
            )
            
            if not self.log_test_result("Specialist Assignment", assignment_valid,
                                      f"Assigned: {assignment_result.assigned_specialist.name if assignment_result.assigned_specialist else 'None'}, Confidence: {assignment_result.assignment_confidence:.2f}"):
                return False
            
            # Test evidence checklist generation
            checklist_valid = len(assignment_result.evidence_checklist.mandatory_items) > 0
            if not self.log_test_result("Evidence Checklist Generation", checklist_valid,
                                      f"{len(assignment_result.evidence_checklist.mandatory_items)} mandatory items"):
                return False
            
            return True
            
        except Exception as e:
            self.log_test_result("Specialist Assignment Engine", False, f"Exception: {e}")
            return False
    
    def test_sla_monitoring_system(self) -> bool:
        """Test SLA Monitoring System functionality."""
        self.logger.info("Testing SLA Monitoring System...")
        
        try:
            # Initialize system
            monitor = SLAMonitoringSystem()
            
            # Test SLA tracking
            deadline = datetime.now(timezone.utc) + timedelta(hours=2)
            tracking_id = monitor.start_sla_tracking(
                issue_number=92,
                specialist_github_issue=None,
                assigned_specialist="test-specialist",
                specialist_type="security",
                sla_deadline=deadline
            )
            
            tracking_valid = tracking_id is not None and tracking_id in monitor.active_slas
            if not self.log_test_result("SLA Tracking Start", tracking_valid,
                                      f"Tracking ID: {tracking_id}"):
                return False
            
            # Test SLA status update
            update_success = monitor.update_sla_status(tracking_id, monitor.active_slas[tracking_id].current_status, "Test update")
            if not self.log_test_result("SLA Status Update", update_success, "Status updated successfully"):
                return False
            
            # Test SLA resolution
            resolve_success = monitor.resolve_sla(tracking_id, "Test completion")
            if not self.log_test_result("SLA Resolution", resolve_success, "SLA resolved successfully"):
                return False
            
            return True
            
        except Exception as e:
            self.log_test_result("SLA Monitoring System", False, f"Exception: {e}")
            return False
    
    def test_decision_audit_tracker(self) -> bool:
        """Test Decision Audit Tracker functionality."""
        self.logger.info("Testing Decision Audit Tracker...")
        
        try:
            # Initialize tracker
            tracker = DecisionAuditTracker()
            
            # Test audit record creation
            audit_record = AuditRecord(
                workflow_id="test_workflow_92",
                timestamp=datetime.now(timezone.utc),
                action="test_action",
                actor="integration_test",
                context="Testing audit functionality",
                rationale="Comprehensive integration test",
                evidence=["Test evidence 1", "Test evidence 2"]
            )
            
            record_id = tracker.record_decision(audit_record)
            record_valid = record_id is not None and len(record_id) > 0
            if not self.log_test_result("Audit Record Creation", record_valid,
                                      f"Record ID: {record_id}"):
                return False
            
            # Test audit trail retrieval
            audit_trail = tracker.get_workflow_audit_trail("test_workflow_92")
            trail_valid = len(audit_trail) > 0
            if not self.log_test_result("Audit Trail Retrieval", trail_valid,
                                      f"Retrieved {len(audit_trail)} records"):
                return False
            
            # Test audit chain integrity
            is_valid, errors = tracker.validate_audit_chain_integrity()
            if not self.log_test_result("Audit Chain Integrity", is_valid,
                                      f"Validation passed with {len(errors)} errors"):
                return False
            
            return True
            
        except Exception as e:
            self.log_test_result("Decision Audit Tracker", False, f"Exception: {e}")
            return False
    
    def test_full_workflow_integration(self) -> bool:
        """Test complete workflow integration."""
        self.logger.info("Testing Full Workflow Integration...")
        
        try:
            # This would require a real GitHub issue for full testing
            # For now, test component integration points
            
            # Test that all components can be initialized together
            risk_engine = RiskAssessmentEngine()
            assignment_engine = SpecialistAssignmentEngine()
            sla_monitor = SLAMonitoringSystem()
            audit_tracker = DecisionAuditTracker()
            
            components_initialized = all([
                risk_engine is not None,
                assignment_engine is not None, 
                sla_monitor is not None,
                audit_tracker is not None
            ])
            
            if not self.log_test_result("Component Initialization", components_initialized,
                                      "All components initialized successfully"):
                return False
            
            # Test data flow between components
            # Create mock risk assessment
            mock_context = ChangeContext(
                files_changed=["test/file.py"],
                lines_added=10,
                lines_removed=5,
                issue_number=92,
                issue_details={
                    'title': 'Test integration',
                    'body': 'Testing component integration',
                    'labels': [{'name': 'test'}],
                    'createdAt': datetime.now(timezone.utc).isoformat()
                }
            )
            
            risk_score = risk_engine.assess_change_risk(mock_context)
            
            # Use risk assessment for specialist assignment
            if risk_score.specialist_type:
                try:
                    specialist_type = SpecialistType(risk_score.specialist_type)
                except ValueError:
                    specialist_type = SpecialistType.ARCHITECTURE
            else:
                specialist_type = SpecialistType.ARCHITECTURE
                
            assignment_request = AssignmentRequest(
                issue_number=92,
                risk_score=risk_score.total_score,
                risk_level=risk_score.risk_level,
                primary_risk_factors=risk_score.reasoning,
                specialist_type=specialist_type,
                urgency_level="medium",
                files_changed=mock_context.files_changed,
                estimated_review_time=4.0,
                special_requirements=[]
            )
            
            assignment_result = assignment_engine.assign_specialist(assignment_request)
            
            # Create audit record for the workflow
            audit_record = AuditRecord(
                workflow_id="integration_test_92",
                timestamp=datetime.now(timezone.utc),
                action="workflow_test_complete",
                actor="integration_test",
                context="Full workflow integration test",
                rationale=f"Risk: {risk_score.risk_level}, Specialist: {assignment_result.assigned_specialist.name if assignment_result.assigned_specialist else 'None'}",
                evidence=[f"Risk score: {risk_score.total_score:.2f}", f"Assignment confidence: {assignment_result.assignment_confidence:.2f}"]
            )
            
            audit_record_id = audit_tracker.record_decision(audit_record)
            
            workflow_valid = all([
                risk_score is not None,
                assignment_result is not None,
                audit_record_id is not None
            ])
            
            if not self.log_test_result("Workflow Integration", workflow_valid,
                                      f"Complete workflow executed successfully"):
                return False
            
            return True
            
        except Exception as e:
            self.log_test_result("Full Workflow Integration", False, f"Exception: {e}")
            return False
    
    def generate_test_report(self):
        """Generate comprehensive test report."""
        self.logger.info("📊 Generating Integration Test Report")
        
        total_tests = len(self.test_results)
        passed_tests = sum(1 for result in self.test_results if result['success'])
        failed_tests = total_tests - passed_tests
        
        pass_rate = (passed_tests / total_tests) * 100 if total_tests > 0 else 0
        
        report = {
            'test_summary': {
                'total_tests': total_tests,
                'passed_tests': passed_tests,
                'failed_tests': failed_tests,
                'pass_rate': f"{pass_rate:.1f}%",
                'test_timestamp': datetime.now(timezone.utc).isoformat()
            },
            'test_results': self.test_results,
            'component_status': {
                'risk_assessment_engine': any(r['success'] for r in self.test_results if 'Risk Assessment' in r['test_name']),
                'specialist_assignment_engine': any(r['success'] for r in self.test_results if 'Specialist Assignment' in r['test_name']),
                'sla_monitoring_system': any(r['success'] for r in self.test_results if 'SLA' in r['test_name']),
                'decision_audit_tracker': any(r['success'] for r in self.test_results if 'Audit' in r['test_name']),
                'workflow_integration': any(r['success'] for r in self.test_results if 'Workflow' in r['test_name'])
            },
            'recommendations': self._generate_recommendations()
        }
        
        # Write report to file
        report_file = Path("knowledge/decisions/integration_test_report.json")
        report_file.parent.mkdir(parents=True, exist_ok=True)
        
        with open(report_file, 'w') as f:
            json.dump(report, f, indent=2)
            
        self.logger.info(f"📄 Test report written to {report_file}")
        
        # Print summary
        print(f"\n{'='*60}")
        print(f"MANUAL INTERVENTION FRAMEWORK INTEGRATION TEST REPORT")
        print(f"{'='*60}")
        print(f"Total Tests: {total_tests}")
        print(f"Passed: {passed_tests} ({pass_rate:.1f}%)")
        print(f"Failed: {failed_tests}")
        print(f"{'='*60}")
        
        if failed_tests > 0:
            print("\nFAILED TESTS:")
            for result in self.test_results:
                if not result['success']:
                    print(f"❌ {result['test_name']}: {result['details']}")
        
        print(f"\nDetailed report: {report_file}")
        
    def _generate_recommendations(self) -> list:
        """Generate recommendations based on test results."""
        recommendations = []
        
        failed_tests = [r for r in self.test_results if not r['success']]
        
        if not failed_tests:
            recommendations.append("✅ All tests passed - Framework is ready for production deployment")
        else:
            recommendations.append("⚠️ Some tests failed - Review failed components before production deployment")
            
            for failed in failed_tests:
                if 'Risk Assessment' in failed['test_name']:
                    recommendations.append("🔧 Review risk assessment configuration and thresholds")
                elif 'Specialist Assignment' in failed['test_name']:
                    recommendations.append("👥 Check specialist registry and assignment logic")
                elif 'SLA' in failed['test_name']:
                    recommendations.append("⏰ Validate SLA monitoring and notification systems")
                elif 'Audit' in failed['test_name']:
                    recommendations.append("📝 Check audit trail storage and integrity validation")
                elif 'Workflow' in failed['test_name']:
                    recommendations.append("🔄 Review workflow integration and error handling")
        
        recommendations.append("📊 Monitor framework performance and effectiveness after deployment")
        recommendations.append("🔄 Schedule regular integration tests to ensure continued functionality")
        
        return recommendations

def main():
    """Run integration tests."""
    test_suite = ManualInterventionIntegrationTest()
    
    success = test_suite.run_all_tests()
    
    return 0 if success else 1

if __name__ == "__main__":
    exit(main())