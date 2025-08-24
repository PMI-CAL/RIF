#!/usr/bin/env python3
"""
Comprehensive Test Suite for Domain Agent Template System - Issue #70
Tests for template validation, capability mapping, and agent initialization
"""

import pytest
import yaml
import json
import tempfile
import shutil
from pathlib import Path
from unittest.mock import Mock, patch, MagicMock
from typing import Dict, Any

# Import the components being tested
import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from claude.commands.template_validator import (
    DomainAgentTemplateValidator,
    ValidationResult,
    ValidationSeverity,
    CapabilityRegistry,
    ExpertiseValidator,
    ValidationIssue
)
from claude.commands.domain_agent_base import (
    DomainAgent,
    AgentTemplate,
    AgentCapability,
    CapabilityMatch,
    AgentConfiguration,
    TaskResult,
    AgentStatus
)

class TestDomainAgentTemplate:
    """Test the AgentTemplate class"""
    
    @pytest.fixture
    def sample_template_data(self):
        """Sample template data for testing"""
        return {
            "domain_agent_template": {
                "metadata": {
                    "name": "test-agent",
                    "version": "1.0.0",
                    "domain": "test",
                    "description": "Test agent template"
                },
                "capabilities": {
                    "primary": ["analysis", "implementation"],
                    "secondary": ["testing"],
                    "cross_domain": ["documentation"]
                },
                "expertise": {
                    "technologies": {
                        "primary": ["python", "javascript"],
                        "secondary": ["docker"]
                    },
                    "patterns": {
                        "design": ["mvc", "factory"]
                    },
                    "domains": {
                        "core": ["web_development"]
                    }
                },
                "tools": {
                    "required": {
                        "development": ["python", "node.js"],
                        "testing": ["pytest", "jest"]
                    }
                },
                "validation_rules": {
                    "code_quality": ["test_coverage_80", "documentation_completeness"]
                },
                "configuration": {
                    "priority_weight": 1.0,
                    "parallel_execution": True
                }
            }
        }
    
    @pytest.fixture
    def sample_template_file(self, sample_template_data):
        """Create a temporary template file for testing"""
        with tempfile.NamedTemporaryFile(mode='w', suffix='.yaml', delete=False) as f:
            yaml.dump(sample_template_data, f)
            temp_path = f.name
        
        yield Path(temp_path)
        
        # Cleanup
        Path(temp_path).unlink(missing_ok=True)
    
    def test_agent_template_from_yaml(self, sample_template_file):
        """Test loading AgentTemplate from YAML file"""
        template = AgentTemplate.from_yaml(sample_template_file)
        
        assert template.metadata["name"] == "test-agent"
        assert template.metadata["domain"] == "test"
        assert template.capabilities["primary"] == ["analysis", "implementation"]
        assert template.expertise["technologies"]["primary"] == ["python", "javascript"]
        assert template.tools["required"]["development"] == ["python", "node.js"]
    
    def test_agent_template_missing_file(self):
        """Test handling of missing template file"""
        with pytest.raises(FileNotFoundError):
            AgentTemplate.from_yaml(Path("nonexistent.yaml"))

class TestCapabilityRegistry:
    """Test the CapabilityRegistry class"""
    
    @pytest.fixture
    def capability_registry(self):
        """Create a CapabilityRegistry instance"""
        return CapabilityRegistry()
    
    def test_valid_capabilities(self, capability_registry):
        """Test validation of known capabilities"""
        assert capability_registry.is_valid_capability("analysis")
        assert capability_registry.is_valid_capability("implementation")
        assert capability_registry.is_valid_capability("ui_development")
    
    def test_invalid_capabilities(self, capability_registry):
        """Test validation of unknown capabilities"""
        assert not capability_registry.is_valid_capability("nonexistent_capability")
        assert not capability_registry.is_valid_capability("invalid_skill")
    
    def test_capability_suggestions(self, capability_registry):
        """Test capability similarity suggestions"""
        suggestions = capability_registry.suggest_similar_capabilities("ui_develop")
        assert "ui_development" in suggestions
        
        suggestions = capability_registry.suggest_similar_capabilities("test")
        assert any("test" in cap for cap in suggestions)
    
    def test_get_capability_info(self, capability_registry):
        """Test getting capability information"""
        info = capability_registry.get_capability_info("analysis")
        assert info is not None
        assert "description" in info
        assert "category" in info

class TestExpertiseValidator:
    """Test the ExpertiseValidator class"""
    
    @pytest.fixture
    def expertise_validator(self):
        """Create an ExpertiseValidator instance"""
        return ExpertiseValidator()
    
    def test_validate_technologies(self, expertise_validator):
        """Test technology validation"""
        known_technologies = ["python", "javascript", "react", "docker"]
        unknown_technologies = ["nonexistent_tech", "invalid_framework"]
        
        unknown_tech = expertise_validator.validate_technologies(known_technologies)
        assert len(unknown_tech) == 0
        
        unknown_tech = expertise_validator.validate_technologies(unknown_technologies)
        assert len(unknown_tech) == 2
    
    def test_validate_patterns(self, expertise_validator):
        """Test pattern validation"""
        known_patterns = ["mvc", "factory", "observer", "singleton"]
        unknown_patterns = ["nonexistent_pattern", "invalid_design"]
        
        unknown_pat = expertise_validator.validate_patterns(known_patterns)
        assert len(unknown_pat) == 0
        
        unknown_pat = expertise_validator.validate_patterns(unknown_patterns)
        assert len(unknown_pat) == 2
    
    def test_validate_domains(self, expertise_validator):
        """Test domain validation"""
        known_domains = ["frontend", "backend", "security", "testing"]
        unknown_domains = ["nonexistent_domain", "invalid_area"]
        
        unknown_dom = expertise_validator.validate_domains(known_domains)
        assert len(unknown_dom) == 0
        
        unknown_dom = expertise_validator.validate_domains(unknown_domains)
        assert len(unknown_dom) == 2

class TestDomainAgentTemplateValidator:
    """Test the DomainAgentTemplateValidator class"""
    
    @pytest.fixture
    def validator(self):
        """Create a DomainAgentTemplateValidator instance"""
        return DomainAgentTemplateValidator()
    
    @pytest.fixture
    def valid_template_file(self):
        """Create a valid template file for testing"""
        template_data = {
            "domain_agent_template": {
                "metadata": {
                    "name": "test-agent",
                    "version": "1.0.0",
                    "domain": "test"
                },
                "capabilities": {
                    "primary": ["analysis"]
                },
                "expertise": {
                    "technologies": {
                        "primary": ["python"]
                    },
                    "patterns": {
                        "design": ["mvc"]
                    },
                    "domains": {
                        "core": ["backend"]
                    }
                },
                "tools": {
                    "required": {
                        "development": ["python"]
                    }
                },
                "validation_rules": {
                    "code_quality": ["test_coverage_80"]
                }
            }
        }
        
        with tempfile.NamedTemporaryFile(mode='w', suffix='.yaml', delete=False) as f:
            yaml.dump(template_data, f)
            temp_path = f.name
        
        yield Path(temp_path)
        
        # Cleanup
        Path(temp_path).unlink(missing_ok=True)
    
    @pytest.fixture
    def invalid_template_file(self):
        """Create an invalid template file for testing"""
        template_data = {
            "domain_agent_template": {
                "metadata": {
                    "name": "test-agent"
                    # Missing required fields
                },
                "capabilities": {
                    # Missing primary capabilities
                }
            }
        }
        
        with tempfile.NamedTemporaryFile(mode='w', suffix='.yaml', delete=False) as f:
            yaml.dump(template_data, f)
            temp_path = f.name
        
        yield Path(temp_path)
        
        # Cleanup
        Path(temp_path).unlink(missing_ok=True)
    
    def test_validate_valid_template(self, validator, valid_template_file):
        """Test validation of a valid template"""
        result = validator.validate_template(valid_template_file)
        
        assert isinstance(result, ValidationResult)
        assert result.valid is True
        assert result.validation_time_ms > 0
    
    def test_validate_invalid_template(self, validator, invalid_template_file):
        """Test validation of an invalid template"""
        result = validator.validate_template(invalid_template_file)
        
        assert isinstance(result, ValidationResult)
        assert result.valid is False
        assert len(result.errors) > 0
    
    def test_validate_nonexistent_file(self, validator):
        """Test validation of nonexistent file"""
        result = validator.validate_template(Path("nonexistent.yaml"))
        
        assert result.valid is False
        assert len(result.errors) > 0
    
    def test_capability_mapping_validation(self, validator):
        """Test capability mapping validation"""
        valid_capabilities = ["analysis", "implementation", "testing"]
        invalid_capabilities = ["nonexistent_capability", "invalid_skill"]
        
        result = validator.validate_capability_mapping(valid_capabilities)
        assert result.valid is True
        
        result = validator.validate_capability_mapping(invalid_capabilities)
        assert result.valid is False
        assert len(result.errors) > 0
    
    def test_validation_report_generation(self, validator, valid_template_file):
        """Test validation report generation"""
        result = validator.validate_template(valid_template_file)
        report = validator.get_validation_report(result)
        
        assert isinstance(report, dict)
        assert "template_path" in report
        assert "validation_summary" in report
        assert "validation_status" in report
        assert "performance_metrics" in report

class MockDomainAgent(DomainAgent):
    """Mock implementation of DomainAgent for testing"""
    
    def execute_primary_task(self, task_data):
        """Mock implementation"""
        return TaskResult(
            task_id="test",
            status=AgentStatus.COMPLETED,
            start_time=None,
            confidence_score=0.9,
            result_data={"success": True}
        )

class TestDomainAgent:
    """Test the DomainAgent base class with template integration"""
    
    @pytest.fixture
    def test_template_file(self):
        """Create a test template file"""
        template_data = {
            "domain_agent_template": {
                "metadata": {
                    "name": "test-agent",
                    "version": "1.0.0",
                    "domain": "test",
                    "description": "Test agent"
                },
                "capabilities": {
                    "primary": ["analysis", "implementation"],
                    "secondary": ["testing"]
                },
                "expertise": {
                    "technologies": {
                        "primary": ["python", "javascript"]
                    },
                    "patterns": {
                        "design": ["mvc"]
                    },
                    "domains": {
                        "core": ["backend"]
                    }
                },
                "tools": {
                    "required": {
                        "development": ["python"]
                    }
                },
                "validation_rules": {
                    "code_quality": ["test_coverage_80"]
                }
            }
        }
        
        with tempfile.NamedTemporaryFile(mode='w', suffix='.yaml', delete=False) as f:
            yaml.dump(template_data, f)
            temp_path = f.name
        
        yield Path(temp_path)
        
        # Cleanup
        Path(temp_path).unlink(missing_ok=True)
    
    @pytest.fixture
    def capability_config_file(self):
        """Create a temporary capability configuration file"""
        config_dir = Path("config")
        config_dir.mkdir(exist_ok=True)
        
        config_data = {
            "capabilities": {
                "development": {
                    "analysis": {
                        "description": "Analyze code and requirements",
                        "complexity": "medium",
                        "estimated_time_minutes": 15
                    },
                    "implementation": {
                        "description": "Implement solutions",
                        "complexity": "high",
                        "estimated_time_minutes": 60
                    }
                },
                "quality": {
                    "testing": {
                        "description": "Design and execute tests",
                        "complexity": "medium",
                        "estimated_time_minutes": 30
                    }
                }
            }
        }
        
        config_path = config_dir / "domain-capabilities.yaml"
        with open(config_path, 'w') as f:
            yaml.dump(config_data, f)
        
        yield config_path
        
        # Cleanup
        config_path.unlink(missing_ok=True)
    
    def test_agent_initialization_with_template(self, test_template_file, capability_config_file):
        """Test agent initialization with template"""
        with patch('claude.commands.template_validator.DomainAgentTemplateValidator') as mock_validator_class:
            # Mock the validator to return a valid result
            mock_validator = Mock()
            mock_result = Mock()
            mock_result.valid = True
            mock_result.errors = []
            mock_result.validation_time_ms = 50.0
            mock_validator.validate_template.return_value = mock_result
            mock_validator_class.return_value = mock_validator
            
            agent = MockDomainAgent(template_path=test_template_file)
            
            assert agent.domain == "test"
            assert agent.name == "test-agent"
            assert len(agent.capabilities) > 0
            assert agent.template is not None
    
    def test_agent_initialization_without_template(self):
        """Test agent initialization without template (fallback mode)"""
        agent = MockDomainAgent(
            domain="test",
            capabilities=["analysis", "testing"],
            name="TestAgent"
        )
        
        assert agent.domain == "test"
        assert agent.name == "TestAgent"
        assert len(agent.capabilities) == 2
        assert agent.template is None
    
    def test_capability_matching(self, test_template_file, capability_config_file):
        """Test capability matching functionality"""
        with patch('claude.commands.template_validator.DomainAgentTemplateValidator') as mock_validator_class:
            # Mock the validator
            mock_validator = Mock()
            mock_result = Mock()
            mock_result.valid = True
            mock_result.errors = []
            mock_result.validation_time_ms = 50.0
            mock_validator.validate_template.return_value = mock_result
            mock_validator_class.return_value = mock_validator
            
            agent = MockDomainAgent(template_path=test_template_file)
            
            # Test task matching
            match_result = agent.can_handle_task("analyze the codebase for issues")
            
            assert isinstance(match_result, CapabilityMatch)
            assert match_result.overall_score >= 0.0
            assert isinstance(match_result.capability_breakdown, dict)
            assert isinstance(match_result.limitations, list)
    
    def test_capability_extraction(self):
        """Test capability extraction from task descriptions"""
        agent = MockDomainAgent(domain="test", capabilities=["analysis"])
        
        # Test various task descriptions
        caps = agent._extract_required_capabilities("analyze the code quality")
        assert "analysis" in caps
        
        caps = agent._extract_required_capabilities("implement a new feature")
        assert "implementation" in caps
        
        caps = agent._extract_required_capabilities("test the application")
        assert "testing" in caps
    
    def test_template_info_retrieval(self, test_template_file, capability_config_file):
        """Test template information retrieval"""
        with patch('claude.commands.template_validator.DomainAgentTemplateValidator') as mock_validator_class:
            # Mock the validator
            mock_validator = Mock()
            mock_result = Mock()
            mock_result.valid = True
            mock_result.errors = []
            mock_result.validation_time_ms = 50.0
            mock_validator.validate_template.return_value = mock_result
            mock_validator_class.return_value = mock_validator
            
            agent = MockDomainAgent(template_path=test_template_file)
            info = agent.get_template_info()
            
            assert info["template_loaded"] is True
            assert "metadata" in info
            assert "performance_metrics" in info
    
    def test_agent_without_template_info(self):
        """Test template info for agent without template"""
        agent = MockDomainAgent(domain="test", capabilities=["analysis"])
        info = agent.get_template_info()
        
        assert info["template_loaded"] is False

class TestTemplateSystemIntegration:
    """Integration tests for the complete template system"""
    
    @pytest.fixture
    def test_environment(self):
        """Set up a complete test environment"""
        # Create temporary directory structure
        temp_dir = Path(tempfile.mkdtemp())
        
        # Create config directory
        config_dir = temp_dir / "config"
        config_dir.mkdir()
        
        # Create templates directory
        templates_dir = temp_dir / "templates"
        templates_dir.mkdir()
        
        # Create capability config
        capability_data = {
            "capabilities": {
                "development": {
                    "analysis": {
                        "description": "Analyze code and requirements",
                        "complexity": "medium",
                        "estimated_time_minutes": 15
                    }
                }
            }
        }
        
        with open(config_dir / "domain-capabilities.yaml", 'w') as f:
            yaml.dump(capability_data, f)
        
        # Create test template
        template_data = {
            "domain_agent_template": {
                "metadata": {
                    "name": "integration-test-agent",
                    "version": "1.0.0",
                    "domain": "test"
                },
                "capabilities": {
                    "primary": ["analysis"]
                },
                "expertise": {
                    "technologies": {
                        "primary": ["python"]
                    },
                    "patterns": {
                        "design": ["mvc"]
                    },
                    "domains": {
                        "core": ["backend"]
                    }
                },
                "tools": {
                    "required": {
                        "development": ["python"]
                    }
                },
                "validation_rules": {
                    "code_quality": ["test_coverage_80"]
                }
            }
        }
        
        template_file = templates_dir / "test-agent.yaml"
        with open(template_file, 'w') as f:
            yaml.dump(template_data, f)
        
        yield {
            "temp_dir": temp_dir,
            "config_dir": config_dir,
            "templates_dir": templates_dir,
            "template_file": template_file
        }
        
        # Cleanup
        shutil.rmtree(temp_dir)
    
    def test_end_to_end_template_workflow(self, test_environment):
        """Test complete end-to-end template workflow"""
        # Change to test directory
        original_cwd = os.getcwd()
        os.chdir(test_environment["temp_dir"])
        
        try:
            # Validate template
            validator = DomainAgentTemplateValidator()
            validation_result = validator.validate_template(test_environment["template_file"])
            
            # Should pass validation
            assert validation_result.valid
            
            # Create agent from template
            with patch('claude.commands.template_validator.DomainAgentTemplateValidator') as mock_validator_class:
                # Mock the validator
                mock_validator = Mock()
                mock_validator.validate_template.return_value = validation_result
                mock_validator_class.return_value = mock_validator
                
                agent = MockDomainAgent(template_path=test_environment["template_file"])
                
                # Verify agent properties
                assert agent.domain == "test"
                assert agent.name == "integration-test-agent"
                assert len(agent.capabilities) > 0
                
                # Test capability matching
                match = agent.can_handle_task("analyze the system architecture")
                assert match.overall_score > 0.5
                
                # Test task execution
                task_result = agent.execute_task("test_task", {"description": "test task"})
                assert task_result.status == AgentStatus.COMPLETED
        
        finally:
            os.chdir(original_cwd)
    
    def test_template_validation_pipeline(self, test_environment):
        """Test the complete template validation pipeline"""
        # Change to test directory for relative path resolution
        original_cwd = os.getcwd()
        os.chdir(test_environment["temp_dir"])
        
        try:
            validator = DomainAgentTemplateValidator()
            
            # Test schema validation
            result = validator.validate_template(test_environment["template_file"])
            assert result.schema_validation is not None
            assert result.schema_validation.yaml_valid
            
            # Test semantic validation
            assert result.semantic_validation is not None
            
            # Test performance validation
            assert result.performance_validation is not None
            assert result.performance_validation.load_time_ms > 0
            
            # Test report generation
            report = validator.get_validation_report(result)
            assert report["validation_status"] in ["PASSED", "FAILED"]
        
        finally:
            os.chdir(original_cwd)

# Performance and stress tests
class TestTemplateSystemPerformance:
    """Performance tests for the template system"""
    
    def test_template_loading_performance(self):
        """Test template loading performance"""
        # Create a large template
        large_template_data = {
            "domain_agent_template": {
                "metadata": {
                    "name": "performance-test-agent",
                    "version": "1.0.0",
                    "domain": "performance"
                },
                "capabilities": {
                    "primary": ["analysis"] * 50,  # Large capability list
                    "secondary": ["testing"] * 30
                },
                "expertise": {
                    "technologies": {
                        "primary": ["python"] * 100  # Large technology list
                    },
                    "patterns": {
                        "design": ["mvc"] * 50
                    },
                    "domains": {
                        "core": ["backend"] * 20
                    }
                },
                "tools": {
                    "required": {
                        "development": ["python"] * 100
                    }
                },
                "validation_rules": {
                    "code_quality": ["test_coverage_80"] * 20
                }
            }
        }
        
        with tempfile.NamedTemporaryFile(mode='w', suffix='.yaml', delete=False) as f:
            yaml.dump(large_template_data, f)
            temp_path = f.name
        
        try:
            import time
            start_time = time.time()
            template = AgentTemplate.from_yaml(Path(temp_path))
            end_time = time.time()
            
            load_time_ms = (end_time - start_time) * 1000
            
            # Should load within reasonable time (< 100ms)
            assert load_time_ms < 100
            assert template.metadata["name"] == "performance-test-agent"
        
        finally:
            Path(temp_path).unlink(missing_ok=True)
    
    def test_validation_performance(self):
        """Test validation performance with multiple templates"""
        validator = DomainAgentTemplateValidator()
        
        # Create multiple template files
        template_files = []
        for i in range(10):
            template_data = {
                "domain_agent_template": {
                    "metadata": {
                        "name": f"test-agent-{i}",
                        "version": "1.0.0",
                        "domain": "test"
                    },
                    "capabilities": {
                        "primary": ["analysis"]
                    },
                    "expertise": {
                        "technologies": {
                            "primary": ["python"]
                        },
                        "patterns": {
                            "design": ["mvc"]
                        },
                        "domains": {
                            "core": ["backend"]
                        }
                    },
                    "tools": {
                        "required": {
                            "development": ["python"]
                        }
                    },
                    "validation_rules": {
                        "code_quality": ["test_coverage_80"]
                    }
                }
            }
            
            with tempfile.NamedTemporaryFile(mode='w', suffix='.yaml', delete=False) as f:
                yaml.dump(template_data, f)
                template_files.append(Path(f.name))
        
        try:
            import time
            start_time = time.time()
            
            results = []
            for template_file in template_files:
                result = validator.validate_template(template_file)
                results.append(result)
            
            end_time = time.time()
            total_time_ms = (end_time - start_time) * 1000
            
            # Should validate all templates quickly
            assert total_time_ms < 1000  # Less than 1 second for 10 templates
            assert all(result.valid for result in results)
        
        finally:
            for template_file in template_files:
                template_file.unlink(missing_ok=True)

if __name__ == "__main__":
    pytest.main([__file__, "-v"])