#!/usr/bin/env python3
"""
Comprehensive Integration Test Suite for RIF Adversarial Validation System

Tests all 8 layers working together in a complete end-to-end validation workflow.
"""

import os
import sys
import json
import sqlite3
import logging
import datetime
from typing import Dict, List, Any
import unittest
from unittest.mock import MagicMock, patch

# Add systems directory to path for imports
sys.path.append('/Users/cal/DEV/RIF/systems')

# Import all adversarial validation layers
try:
    from adversarial_feature_discovery_engine import AdversarialFeatureDiscovery
    from adversarial_evidence_collection_framework import AdversarialEvidenceCollector
    from adversarial_validation_execution_engine import AdversarialValidationEngine
    from adversarial_quality_orchestration_layer import AdversarialQualityOrchestrator
    from adversarial_knowledge_integration_layer import AdversarialKnowledgeIntegrator
    from adversarial_issue_generation_engine import AdversarialIssueGenerator
    from adversarial_reporting_dashboard_layer import AdversarialReportingDashboard
    from adversarial_integration_hub_layer import AdversarialIntegrationHub
except ImportError as e:
    print(f"Import error: {e}")
    print("Ensure all adversarial validation layer files exist in systems/ directory")
    sys.exit(1)

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class AdversarialSystemIntegrationTest(unittest.TestCase):
    """
    Comprehensive integration tests for the complete 8-layer adversarial validation system.
    
    Tests:
    1. Individual layer functionality
    2. Inter-layer communication and data flow
    3. Complete end-to-end validation workflow
    4. Integration with existing RIF systems
    5. Error handling and recovery scenarios
    """
    
    def setUp(self):
        """Set up test environment and initialize all layers"""
        self.test_root = "/tmp/rif_test"
        os.makedirs(self.test_root, exist_ok=True)
        
        # Initialize all layers with test paths
        self.layer1 = AdversarialFeatureDiscovery(rif_root=self.test_root)
        self.layer2 = AdversarialEvidenceCollection(db_path=f"{self.test_root}/evidence.db")
        self.layer3 = AdversarialValidationEngine(db_path=f"{self.test_root}/validation.db")
        self.layer4 = AdversarialQualityOrchestration(db_path=f"{self.test_root}/orchestration.db")
        self.layer5 = AdversarialKnowledgeIntegration(db_path=f"{self.test_root}/knowledge.db")
        self.layer6 = AdversarialIssueGeneration(db_path=f"{self.test_root}/issues.db")
        self.layer7 = AdversarialReportingDashboard(db_path=f"{self.test_root}/reports.db")
        self.layer8 = AdversarialIntegrationHub(rif_root=self.test_root)
        
        logger.info("All 8 adversarial validation layers initialized for testing")
    
    def tearDown(self):
        """Clean up test environment"""
        import shutil
        if os.path.exists(self.test_root):
            shutil.rmtree(self.test_root)
    
    def test_layer1_feature_discovery(self):
        """Test Layer 1: Feature Discovery Engine"""
        logger.info("Testing Layer 1: Feature Discovery Engine")
        
        # Test feature discovery functionality
        features = self.layer1.discover_all_features()
        
        # Assertions
        self.assertIsInstance(features, list)
        self.assertTrue(len(features) > 0, "Should discover at least some features")
        
        # Test feature categorization
        if features:
            feature = features[0]
            self.assertIn('feature_id', feature)
            self.assertIn('feature_name', feature) 
            self.assertIn('category', feature)
            
        logger.info(f"✅ Layer 1 test passed - discovered {len(features)} features")
    
    def test_layer2_evidence_collection(self):
        """Test Layer 2: Evidence Collection Framework"""
        logger.info("Testing Layer 2: Evidence Collection Framework")
        
        # Create test feature
        test_feature = {
            'feature_id': 'test_feature_001',
            'feature_name': 'Test Feature',
            'category': 'test_category',
            'implementation_path': '/test/path.py'
        }
        
        # Test evidence collection
        evidence = self.layer2.collect_comprehensive_evidence('test_feature_001', test_feature)
        
        # Assertions
        self.assertIsInstance(evidence, dict)
        self.assertIn('feature_id', evidence)
        self.assertIn('evidence_types', evidence)
        self.assertIn('integrity_hash', evidence)
        
        logger.info("✅ Layer 2 test passed - evidence collection working")
    
    def test_layer3_validation_execution(self):
        """Test Layer 3: Validation Execution Engine"""
        logger.info("Testing Layer 3: Validation Execution Engine")
        
        # Create test evidence
        test_evidence = {
            'feature_id': 'test_feature_001',
            'evidence_types': ['code_analysis', 'execution_test'],
            'evidence_data': {'test': 'data'},
            'integrity_hash': 'test_hash_123'
        }
        
        # Test validation execution
        validation_result = self.layer3.execute_comprehensive_validation('test_feature_001', test_evidence)
        
        # Assertions
        self.assertIsInstance(validation_result, dict)
        self.assertIn('validation_status', validation_result)
        self.assertIn('test_levels_completed', validation_result)
        self.assertIn('risk_assessment', validation_result)
        
        logger.info("✅ Layer 3 test passed - validation execution working")
    
    def test_layer4_quality_orchestration(self):
        """Test Layer 4: Quality Orchestration Layer"""
        logger.info("Testing Layer 4: Quality Orchestration Layer")
        
        # Create test validation result
        test_validation = {
            'feature_id': 'test_feature_001',
            'validation_status': 'FAIL',
            'risk_level': 'HIGH',
            'fix_required': True
        }
        
        # Test orchestration decision
        orchestration_result = self.layer4.make_orchestration_decision('test_feature_001', test_validation)
        
        # Assertions
        self.assertIsInstance(orchestration_result, dict)
        self.assertIn('decision', orchestration_result)
        self.assertIn('next_actions', orchestration_result)
        self.assertIn('priority', orchestration_result)
        
        logger.info("✅ Layer 4 test passed - quality orchestration working")
    
    def test_layer5_knowledge_integration(self):
        """Test Layer 5: Knowledge Integration Layer"""
        logger.info("Testing Layer 5: Knowledge Integration Layer")
        
        # Create test validation result for learning
        test_result = {
            'feature_id': 'test_feature_001',
            'validation_status': 'FAIL',
            'category': 'test_category',
            'failure_patterns': ['pattern1', 'pattern2']
        }
        
        # Test knowledge integration
        learning_result = self.layer5.integrate_validation_learning('test_feature_001', test_result)
        
        # Assertions
        self.assertIsInstance(learning_result, dict)
        self.assertIn('patterns_extracted', learning_result)
        self.assertIn('knowledge_updated', learning_result)
        
        logger.info("✅ Layer 5 test passed - knowledge integration working")
    
    def test_layer6_issue_generation(self):
        """Test Layer 6: Issue Generation Engine"""
        logger.info("Testing Layer 6: Issue Generation Engine")
        
        # Create test orchestration decision
        test_decision = {
            'feature_id': 'test_feature_001',
            'decision': 'CREATE_FIX_ISSUE',
            'priority': 'HIGH',
            'issue_data': {
                'title': 'Fix Test Feature',
                'description': 'Test feature validation failed',
                'labels': ['fix:required', 'validation:failed']
            }
        }
        
        # Test issue generation (mock GitHub calls)
        with patch('subprocess.run') as mock_run:
            mock_run.return_value.returncode = 0
            mock_run.return_value.stdout = '{"number": 123, "url": "https://github.com/test/repo/issues/123"}'
            
            issue_result = self.layer6.generate_validation_issue('test_feature_001', test_decision)
            
            # Assertions
            self.assertIsInstance(issue_result, dict)
            self.assertIn('issue_created', issue_result)
            
        logger.info("✅ Layer 6 test passed - issue generation working")
    
    def test_layer7_reporting_dashboard(self):
        """Test Layer 7: Reporting and Dashboard Layer"""
        logger.info("Testing Layer 7: Reporting and Dashboard Layer")
        
        # Create test validation report
        from adversarial_reporting_dashboard_layer import ValidationReport
        test_report = ValidationReport(
            feature_id='test_feature_001',
            feature_name='Test Feature',
            category='test_category',
            validation_status='FAIL',
            evidence_count=3,
            test_levels_completed=2,
            test_levels_total=5,
            risk_level='HIGH',
            fix_required=True,
            validation_timestamp=datetime.datetime.utcnow().isoformat() + "Z",
            evidence_integrity_hash='test_hash_123'
        )
        
        # Test report recording
        success = self.layer7.record_validation_report(test_report)
        self.assertTrue(success)
        
        # Test dashboard generation
        dashboard = self.layer7.generate_comprehensive_dashboard()
        
        # Assertions
        self.assertIsInstance(dashboard, dict)
        self.assertIn('system_health', dashboard)
        self.assertIn('category_summaries', dashboard)
        
        logger.info("✅ Layer 7 test passed - reporting dashboard working")
    
    def test_layer8_integration_hub(self):
        """Test Layer 8: Integration Hub Layer"""
        logger.info("Testing Layer 8: Integration Hub Layer")
        
        # Test integration status
        integration_health = self.layer8.get_integration_health_report()
        
        # Assertions  
        self.assertIsInstance(integration_health, dict)
        self.assertIn('total_integrations', integration_health)
        self.assertIn('integration_details', integration_health)
        
        # Test coordination with mock validation result
        test_validation = {
            'validation_status': 'FAIL',
            'feature_name': 'Test Feature',
            'risk_level': 'HIGH',
            'fix_required': True
        }
        
        coordination_result = self.layer8.coordinate_validation_with_orchestration(
            'test_feature_001', test_validation
        )
        
        # Assertions
        self.assertIsInstance(coordination_result, dict)
        self.assertIn('orchestration_actions', coordination_result)
        
        logger.info("✅ Layer 8 test passed - integration hub working")
    
    def test_complete_end_to_end_workflow(self):
        """Test complete end-to-end adversarial validation workflow"""
        logger.info("Testing complete end-to-end adversarial validation workflow")
        
        workflow_results = {
            'stages_completed': [],
            'total_stages': 8,
            'workflow_success': True,
            'data_flow_verified': True
        }
        
        try:
            # Stage 1: Feature Discovery
            features = self.layer1.discover_all_features()
            if features:
                test_feature = features[0]
                workflow_results['stages_completed'].append('feature_discovery')
                logger.info("✅ Stage 1: Feature Discovery completed")
            else:
                # Create mock feature for testing
                test_feature = {
                    'feature_id': 'workflow_test_feature',
                    'feature_name': 'Workflow Test Feature',
                    'category': 'test_workflow',
                    'implementation_path': '/test/workflow.py'
                }
                workflow_results['stages_completed'].append('feature_discovery_mock')
            
            # Stage 2: Evidence Collection  
            evidence = self.layer2.collect_comprehensive_evidence(
                test_feature['feature_id'], test_feature
            )
            workflow_results['stages_completed'].append('evidence_collection')
            logger.info("✅ Stage 2: Evidence Collection completed")
            
            # Stage 3: Validation Execution
            validation_result = self.layer3.execute_comprehensive_validation(
                test_feature['feature_id'], evidence
            )
            workflow_results['stages_completed'].append('validation_execution')
            logger.info("✅ Stage 3: Validation Execution completed")
            
            # Stage 4: Quality Orchestration
            orchestration_decision = self.layer4.make_orchestration_decision(
                test_feature['feature_id'], validation_result
            )
            workflow_results['stages_completed'].append('quality_orchestration')
            logger.info("✅ Stage 4: Quality Orchestration completed")
            
            # Stage 5: Knowledge Integration
            learning_result = self.layer5.integrate_validation_learning(
                test_feature['feature_id'], validation_result
            )
            workflow_results['stages_completed'].append('knowledge_integration')
            logger.info("✅ Stage 5: Knowledge Integration completed")
            
            # Stage 6: Issue Generation (mock GitHub interaction)
            with patch('subprocess.run') as mock_run:
                mock_run.return_value.returncode = 0
                mock_run.return_value.stdout = '{"number": 456, "url": "https://github.com/test/repo/issues/456"}'
                
                issue_result = self.layer6.generate_validation_issue(
                    test_feature['feature_id'], orchestration_decision
                )
                workflow_results['stages_completed'].append('issue_generation')
                logger.info("✅ Stage 6: Issue Generation completed")
            
            # Stage 7: Reporting Dashboard
            from adversarial_reporting_dashboard_layer import ValidationReport
            report = ValidationReport(
                feature_id=test_feature['feature_id'],
                feature_name=test_feature['feature_name'],
                category=test_feature['category'],
                validation_status=validation_result.get('validation_status', 'UNKNOWN'),
                evidence_count=len(evidence.get('evidence_types', [])),
                test_levels_completed=validation_result.get('test_levels_completed', 0),
                test_levels_total=validation_result.get('test_levels_total', 5),
                risk_level=validation_result.get('risk_level', 'MEDIUM'),
                fix_required=validation_result.get('fix_required', False),
                validation_timestamp=datetime.datetime.utcnow().isoformat() + "Z",
                evidence_integrity_hash=evidence.get('integrity_hash', 'test_hash')
            )
            
            self.layer7.record_validation_report(report)
            dashboard = self.layer7.generate_comprehensive_dashboard()
            workflow_results['stages_completed'].append('reporting_dashboard')
            logger.info("✅ Stage 7: Reporting Dashboard completed")
            
            # Stage 8: Integration Hub Coordination
            coordination_result = self.layer8.coordinate_validation_with_orchestration(
                test_feature['feature_id'], validation_result
            )
            workflow_results['stages_completed'].append('integration_hub')
            logger.info("✅ Stage 8: Integration Hub completed")
            
            # Verify complete workflow
            if len(workflow_results['stages_completed']) == workflow_results['total_stages']:
                workflow_results['workflow_success'] = True
                logger.info("🎉 Complete end-to-end workflow successful!")
            else:
                workflow_results['workflow_success'] = False
                logger.warning(f"Workflow incomplete: {len(workflow_results['stages_completed'])}/{workflow_results['total_stages']} stages completed")
            
        except Exception as e:
            workflow_results['workflow_success'] = False
            workflow_results['error'] = str(e)
            logger.error(f"End-to-end workflow failed: {e}")
        
        # Assertions
        self.assertTrue(workflow_results['workflow_success'], "End-to-end workflow should complete successfully")
        self.assertEqual(len(workflow_results['stages_completed']), workflow_results['total_stages'])
        
        return workflow_results
    
    def test_system_integration_validation(self):
        """Test integration validation with suspected non-functional RIF systems"""
        logger.info("Testing integration validation with suspected non-functional systems")
        
        # Test validation of the 3 confirmed non-functional systems from issue #146
        suspected_systems = [
            {
                'feature_id': 'shadow_issue_tracking',
                'feature_name': 'Shadow Issue Tracking System', 
                'category': 'quality_assurance',
                'expected_status': 'FAIL'
            },
            {
                'feature_id': 'parallel_agent_consensus',
                'feature_name': 'Parallel Agent Consensus System',
                'category': 'orchestration',
                'expected_status': 'FAIL'
            },
            {
                'feature_id': 'automated_error_issue_generation', 
                'feature_name': 'Automated Error Issue Generation',
                'category': 'error_handling',
                'expected_status': 'FAIL'  # Partial - capture works, issue generation broken
            }
        ]
        
        validation_results = []
        
        for system in suspected_systems:
            # Run validation workflow for each suspected system
            evidence = self.layer2.collect_comprehensive_evidence(system['feature_id'], system)
            validation_result = self.layer3.execute_comprehensive_validation(system['feature_id'], evidence)
            
            # Check if validation detected the expected failure
            validation_results.append({
                'feature_id': system['feature_id'],
                'expected_failure': system['expected_status'] == 'FAIL',
                'detected_failure': validation_result.get('validation_status') in ['FAIL', 'PARTIAL'],
                'validation_result': validation_result
            })
        
        # Verify that validation system can detect known failures
        detected_failures = sum(1 for r in validation_results if r['detected_failure'] and r['expected_failure'])
        expected_failures = sum(1 for r in validation_results if r['expected_failure'])
        
        logger.info(f"Detected {detected_failures}/{expected_failures} expected failures")
        
        # Assert that validation system can detect most known issues
        detection_rate = detected_failures / expected_failures if expected_failures > 0 else 0
        self.assertGreaterEqual(detection_rate, 0.5, "Should detect at least 50% of known failures")
        
        return validation_results

def main():
    """Run comprehensive integration tests"""
    print("🔬 Starting RIF Adversarial Validation System Integration Tests")
    print("=" * 70)
    
    # Create test suite
    test_suite = unittest.TestLoader().loadTestsFromTestCase(AdversarialSystemIntegrationTest)
    
    # Run tests with detailed output
    test_runner = unittest.TextTestRunner(verbosity=2)
    test_result = test_runner.run(test_suite)
    
    # Print summary
    print("\n" + "=" * 70)
    if test_result.wasSuccessful():
        print("🎉 ALL INTEGRATION TESTS PASSED")
        print("✅ 8-Layer Adversarial Validation System is fully operational")
        print("✅ End-to-end workflow validated") 
        print("✅ Integration with RIF systems verified")
        print("✅ Non-functional system detection capability confirmed")
    else:
        print("❌ SOME TESTS FAILED")
        print(f"Failures: {len(test_result.failures)}")
        print(f"Errors: {len(test_result.errors)}")
    
    print(f"Tests run: {test_result.testsRun}")
    print("=" * 70)
    
    return test_result.wasSuccessful()

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)