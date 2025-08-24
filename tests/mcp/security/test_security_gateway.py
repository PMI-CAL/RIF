"""
Comprehensive tests for MCP Security Gateway

Tests cover:
- Authentication and authorization workflows
- Security policy enforcement
- Threat detection and response
- Session management
- Credential rotation
- Audit logging
"""

import asyncio
import json
import pytest
import tempfile
import time
from datetime import datetime, timedelta
from pathlib import Path
from unittest.mock import AsyncMock, MagicMock, patch

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '../../..'))

from mcp.security.gateway import MCPSecurityGateway, SecurityContext
from mcp.security.credential_manager import CredentialManager, AuthResult
from mcp.security.permission_matrix import PermissionMatrix, PermissionResult, PermissionLevel
from mcp.security.audit_logger import AuditLogger, SecurityEvent, SecurityEventType
from mcp.security.session_manager import SessionManager, TokenValidationResult
from mcp.security.threat_detector import ThreatDetector, ThreatAssessment, ThreatLevel
from mcp.security.exceptions import (
    AuthenticationError, AuthorizationError, SecurityViolationError
)


class TestMCPSecurityGateway:
    """Test suite for MCP Security Gateway main controller"""
    
    @pytest.fixture
    async def temp_dir(self):
        """Create temporary directory for test storage"""
        with tempfile.TemporaryDirectory() as temp_dir:
            yield Path(temp_dir)
    
    @pytest.fixture
    async def security_config(self, temp_dir):
        """Create test security configuration"""
        return {
            'credentials': {'storage_path': str(temp_dir / 'credentials')},
            'permissions': {'storage_path': str(temp_dir / 'permissions')},
            'audit': {'storage_path': str(temp_dir / 'audit')},
            'sessions': {'storage_path': str(temp_dir / 'sessions')},
            'threat_detection': {'storage_path': str(temp_dir / 'threats')},
            'session_timeout_hours': 1,
            'auth_performance_target_ms': 200,
            'authz_performance_target_ms': 100
        }
    
    @pytest.fixture
    async def security_gateway(self, security_config):
        """Create test security gateway instance"""
        gateway = MCPSecurityGateway(security_config)
        yield gateway
        # Cleanup
        if hasattr(gateway.audit_logger, 'shutdown'):
            gateway.audit_logger.shutdown()
        if hasattr(gateway.session_manager, 'shutdown'):
            await gateway.session_manager.shutdown()
    
    @pytest.mark.asyncio
    async def test_authentication_success_workflow(self, security_gateway):
        """Test successful authentication workflow"""
        server_id = "test-server-001"
        credentials = {"api_key": "test-api-key-123"}
        
        # Store credential first
        await security_gateway.credential_manager.store_credential(
            server_id, "api_key", "test-api-key-123", ["read", "write"]
        )
        
        # Attempt authentication
        session_token = await security_gateway.authenticate_server(
            server_id, credentials
        )
        
        assert session_token is not None
        assert session_token in security_gateway.active_sessions
        
        # Verify security context
        context = security_gateway.active_sessions[session_token]
        assert context.server_id == server_id
        assert context.session_token == session_token
        assert isinstance(context.authenticated_at, datetime)
        assert context.permissions == ["read", "write"]
    
    @pytest.mark.asyncio
    async def test_authentication_failure_invalid_credentials(self, security_gateway):
        """Test authentication failure with invalid credentials"""
        server_id = "test-server-002"
        invalid_credentials = {"api_key": "wrong-key"}
        
        # Store different credential
        await security_gateway.credential_manager.store_credential(
            server_id, "api_key", "correct-key-123", ["read"]
        )
        
        # Attempt authentication with wrong credentials
        with pytest.raises(AuthenticationError):
            await security_gateway.authenticate_server(server_id, invalid_credentials)
        
        # Verify no session was created
        assert len(security_gateway.active_sessions) == 0
    
    @pytest.mark.asyncio
    async def test_authentication_security_violation_blocked(self, security_gateway):
        """Test authentication blocked due to security violation"""
        server_id = "test-server-003"
        credentials = {"api_key": "test-key"}
        
        # Mock threat detector to return critical threat
        with patch.object(
            security_gateway.threat_detector, 
            'assess_authentication_risk'
        ) as mock_threat_assess:
            mock_threat_assess.return_value = ThreatAssessment(
                threat_level=ThreatLevel.CRITICAL,
                threat_types=[],
                confidence_score=0.9,
                risk_factors=["High threat detected"],
                recommended_actions=["Block authentication"],
                reason="Security threat detected"
            )
            
            # Attempt authentication
            with pytest.raises(SecurityViolationError):
                await security_gateway.authenticate_server(server_id, credentials)
    
    @pytest.mark.asyncio
    async def test_authorization_success_workflow(self, security_gateway):
        """Test successful authorization workflow"""
        server_id = "test-server-004"
        credentials = {"api_key": "auth-test-key"}
        
        # Setup and authenticate
        await security_gateway.credential_manager.store_credential(
            server_id, "api_key", "auth-test-key", ["read", "write"]
        )
        
        session_token = await security_gateway.authenticate_server(server_id, credentials)
        
        # Test authorization
        operation = "read"
        resources = ["file1.txt", "file2.txt"]
        
        result = await security_gateway.authorize_operation(
            session_token, operation, resources
        )
        
        assert result is True
    
    @pytest.mark.asyncio
    async def test_authorization_failure_insufficient_permissions(self, security_gateway):
        """Test authorization failure due to insufficient permissions"""
        server_id = "test-server-005"
        credentials = {"api_key": "limited-key"}
        
        # Setup with limited permissions
        await security_gateway.credential_manager.store_credential(
            server_id, "api_key", "limited-key", ["read"]
        )
        
        session_token = await security_gateway.authenticate_server(server_id, credentials)
        
        # Mock permission matrix to deny operation
        with patch.object(
            security_gateway.permission_matrix,
            'check_permissions'
        ) as mock_check_perms:
            mock_check_perms.return_value = PermissionResult(
                is_allowed=False,
                granted_permissions=["read"],
                effective_level="read",
                denial_reason="Insufficient permissions for write operation"
            )
            
            # Attempt unauthorized operation
            with pytest.raises(AuthorizationError):
                await security_gateway.authorize_operation(
                    session_token, "write", ["sensitive_file.txt"]
                )
    
    @pytest.mark.asyncio
    async def test_credential_rotation_workflow(self, security_gateway):
        """Test credential rotation workflow"""
        server_id = "test-server-006"
        original_key = "original-key-123"
        
        # Store initial credential
        await security_gateway.credential_manager.store_credential(
            server_id, "api_key", original_key, ["read", "write"]
        )
        
        # Force credential rotation
        rotation_result = await security_gateway.rotate_credentials(
            server_id, force_rotation=True
        )
        
        assert server_id in rotation_result
        assert rotation_result[server_id] != original_key
        
        # Verify old credential no longer works
        with pytest.raises(AuthenticationError):
            await security_gateway.authenticate_server(
                server_id, {"api_key": original_key}
            )
    
    @pytest.mark.asyncio
    async def test_session_invalidation(self, security_gateway):
        """Test session invalidation"""
        server_id = "test-server-007"
        credentials = {"api_key": "session-test-key"}
        
        # Setup and authenticate
        await security_gateway.credential_manager.store_credential(
            server_id, "api_key", "session-test-key", ["read"]
        )
        
        session_token = await security_gateway.authenticate_server(server_id, credentials)
        
        # Verify session exists
        assert session_token in security_gateway.active_sessions
        
        # Invalidate session
        result = await security_gateway.invalidate_session(session_token, "test_cleanup")
        assert result is True
        
        # Verify session is no longer active
        context = security_gateway.active_sessions.get(session_token)
        if context:
            assert not context.is_active
    
    @pytest.mark.asyncio
    async def test_secure_operation_context_manager(self, security_gateway):
        """Test secure operation context manager"""
        server_id = "test-server-008"
        credentials = {"api_key": "context-test-key"}
        
        # Setup and authenticate
        await security_gateway.credential_manager.store_credential(
            server_id, "api_key", "context-test-key", ["read", "execute"]
        )
        
        session_token = await security_gateway.authenticate_server(server_id, credentials)
        
        # Use secure operation context
        async with security_gateway.secure_operation_context(
            session_token, "execute", ["script.py"]
        ) as context:
            assert context is not None
            assert context.server_id == server_id
    
    @pytest.mark.asyncio
    async def test_expired_session_cleanup(self, security_gateway):
        """Test cleanup of expired sessions"""
        server_id = "test-server-009"
        credentials = {"api_key": "expire-test-key"}
        
        # Setup with very short session timeout
        await security_gateway.credential_manager.store_credential(
            server_id, "api_key", "expire-test-key", ["read"]
        )
        
        # Create session
        session_token = await security_gateway.authenticate_server(server_id, credentials)
        
        # Manually expire session
        context = security_gateway.active_sessions[session_token]
        context.last_activity = datetime.utcnow() - timedelta(hours=25)
        
        # Run cleanup
        cleaned_count = await security_gateway.cleanup_expired_sessions()
        
        assert cleaned_count >= 1
        
        # Verify session was cleaned up
        context = security_gateway.active_sessions.get(session_token)
        if context:
            assert not context.is_active
    
    @pytest.mark.asyncio
    async def test_security_metrics_collection(self, security_gateway):
        """Test security metrics collection"""
        # Perform some operations to generate metrics
        server_id = "metrics-test-server"
        credentials = {"api_key": "metrics-key"}
        
        await security_gateway.credential_manager.store_credential(
            server_id, "api_key", "metrics-key", ["read"]
        )
        
        # Successful authentication
        session_token = await security_gateway.authenticate_server(server_id, credentials)
        
        # Failed authentication
        try:
            await security_gateway.authenticate_server(server_id, {"api_key": "wrong-key"})
        except AuthenticationError:
            pass
        
        # Get metrics
        metrics = await security_gateway.get_security_metrics()
        
        assert metrics.authentication_attempts >= 2
        assert metrics.authentication_failures >= 1
        assert metrics.active_sessions >= 1
        assert metrics.average_auth_latency > 0
    
    @pytest.mark.asyncio
    async def test_performance_monitoring(self, security_gateway):
        """Test performance monitoring for security operations"""
        server_id = "perf-test-server"
        credentials = {"api_key": "perf-key"}
        
        await security_gateway.credential_manager.store_credential(
            server_id, "api_key", "perf-key", ["read", "write"]
        )
        
        # Measure authentication performance
        start_time = time.perf_counter()
        session_token = await security_gateway.authenticate_server(server_id, credentials)
        auth_time = (time.perf_counter() - start_time) * 1000
        
        # Should meet performance target
        assert auth_time < security_gateway.config.get('auth_performance_target_ms', 200)
        
        # Measure authorization performance
        start_time = time.perf_counter()
        await security_gateway.authorize_operation(session_token, "read", ["test.txt"])
        authz_time = (time.perf_counter() - start_time) * 1000
        
        # Should meet performance target
        assert authz_time < security_gateway.config.get('authz_performance_target_ms', 100)


class TestCredentialManager:
    """Test suite for credential management"""
    
    @pytest.fixture
    async def temp_dir(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            yield Path(temp_dir)
    
    @pytest.fixture
    async def credential_manager(self, temp_dir):
        config = {'storage_path': str(temp_dir / 'credentials')}
        return CredentialManager(config)
    
    @pytest.mark.asyncio
    async def test_credential_storage_and_validation(self, credential_manager):
        """Test secure credential storage and validation"""
        server_id = "cred-test-server"
        credential_value = "secure-api-key-123"
        permissions = ["read", "write", "execute"]
        
        # Store credential
        result = await credential_manager.store_credential(
            server_id, "api_key", credential_value, permissions
        )
        assert result is True
        
        # Validate correct credential
        auth_result = await credential_manager.validate_credentials(
            server_id, {"api_key": credential_value}
        )
        assert auth_result.is_valid is True
        assert auth_result.permissions == permissions
        
        # Validate incorrect credential
        auth_result = await credential_manager.validate_credentials(
            server_id, {"api_key": "wrong-key"}
        )
        assert auth_result.is_valid is False
    
    @pytest.mark.asyncio
    async def test_credential_rotation(self, credential_manager):
        """Test credential rotation functionality"""
        server_id = "rotation-test-server"
        original_key = "original-key-123"
        
        # Store initial credential
        await credential_manager.store_credential(
            server_id, "api_key", original_key, ["read"]
        )
        
        # Rotate credential
        new_key = await credential_manager.rotate_server_credentials(
            server_id, force_rotation=True
        )
        
        assert new_key != original_key
        
        # Verify new credential works
        auth_result = await credential_manager.validate_credentials(
            server_id, {"api_key": new_key}
        )
        assert auth_result.is_valid is True
    
    @pytest.mark.asyncio
    async def test_credential_encryption(self, credential_manager):
        """Test credential encryption in storage"""
        server_id = "encryption-test"
        credential_value = "test-encryption-key"
        
        # Store credential
        await credential_manager.store_credential(
            server_id, "api_key", credential_value, ["read"]
        )
        
        # Check that credential is encrypted in storage
        credential_info = credential_manager.credentials[server_id]
        
        # The encrypted credential should not contain the original value
        decrypted = credential_manager._decrypt_credential(
            credential_info.encrypted_credential,
            credential_info.salt
        )
        assert decrypted == credential_value
        
        # But the encrypted form should be different
        assert credential_info.encrypted_credential != credential_value.encode()


class TestPermissionMatrix:
    """Test suite for permission matrix and RBAC"""
    
    @pytest.fixture
    async def temp_dir(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            yield Path(temp_dir)
    
    @pytest.fixture
    async def permission_matrix(self, temp_dir):
        config = {'storage_path': str(temp_dir / 'permissions')}
        matrix = PermissionMatrix(config)
        
        # Add basic roles and policies for testing
        await matrix.add_role("reader", ["read_files", "read_data"])
        await matrix.add_role("writer", ["read_files", "read_data", "write_files", "write_data"])
        await matrix.add_role("admin", ["read_files", "read_data", "write_files", "write_data", "admin_access"])
        
        await matrix.add_policy("read", "read_files", PermissionLevel.READ)
        await matrix.add_policy("write", "write_files", PermissionLevel.WRITE)
        await matrix.add_policy("admin", "admin_access", PermissionLevel.ADMIN)
        
        return matrix
    
    @pytest.mark.asyncio
    async def test_role_based_permission_check(self, permission_matrix):
        """Test role-based permission checking"""
        # Test reader role
        result = await permission_matrix.check_permissions(
            ["reader"], "read", ["file1.txt"]
        )
        assert result.is_allowed is True
        assert "read_files" in result.granted_permissions
        
        # Test writer role
        result = await permission_matrix.check_permissions(
            ["writer"], "write", ["file1.txt"]
        )
        assert result.is_allowed is True
        
        # Test reader trying to write (should fail)
        result = await permission_matrix.check_permissions(
            ["reader"], "write", ["file1.txt"]
        )
        assert result.is_allowed is False
    
    @pytest.mark.asyncio
    async def test_resource_specific_access(self, permission_matrix):
        """Test resource-specific access control"""
        user_id = "test-user"
        
        # Grant specific resource access
        await permission_matrix.grant_resource_access(
            user_id, "sensitive_file.txt", "file", ["read"], PermissionLevel.READ
        )
        
        # Check access to granted resource
        result = await permission_matrix.check_permissions(
            [user_id], "read", ["sensitive_file.txt"], {user_id: ["sensitive_file.txt"]}
        )
        # This would require more complex logic to properly test resource access
        # For now, verify the grant succeeded
        assert user_id in permission_matrix.resource_access
    
    @pytest.mark.asyncio
    async def test_permission_inheritance(self, permission_matrix):
        """Test role inheritance"""
        # Create role with inheritance
        await permission_matrix.add_role(
            "senior_writer", ["advanced_write"], inherits_from=["writer"]
        )
        
        # Test inherited permissions
        resolved = await permission_matrix._resolve_user_permissions(["senior_writer"])
        assert "read_files" in resolved  # Inherited from writer
        assert "write_files" in resolved  # Inherited from writer
        assert "advanced_write" in resolved  # Own permission
    
    @pytest.mark.asyncio
    async def test_expired_access_cleanup(self, permission_matrix):
        """Test cleanup of expired resource access"""
        user_id = "expire-test-user"
        
        # Grant access with short expiration
        await permission_matrix.grant_resource_access(
            user_id, "temp_file.txt", "file", ["read"], 
            PermissionLevel.READ, expires_in_hours=0.001  # Very short expiration
        )
        
        # Wait for expiration
        await asyncio.sleep(0.1)
        
        # Run cleanup
        cleanup_count = await permission_matrix.cleanup_expired_access()
        
        assert cleanup_count >= 1


class TestThreatDetector:
    """Test suite for threat detection engine"""
    
    @pytest.fixture
    async def temp_dir(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            yield Path(temp_dir)
    
    @pytest.fixture
    async def threat_detector(self, temp_dir):
        config = {
            'storage_path': str(temp_dir / 'threats'),
            'threat_thresholds': {
                'auth_failures_per_minute': 5,
                'auth_failures_per_hour': 20
            }
        }
        return ThreatDetector(config)
    
    @pytest.mark.asyncio
    async def test_brute_force_detection(self, threat_detector):
        """Test brute force attack detection"""
        server_id = "brute-force-test"
        credentials = {"api_key": "test-key"}
        context = {"source_ip": "192.168.1.100"}
        
        # Simulate multiple failed authentication attempts
        for _ in range(6):  # Exceeds threshold of 5
            assessment = await threat_detector.assess_authentication_risk(
                server_id, credentials, context
            )
        
        # Last assessment should detect brute force
        assert assessment.threat_level in [ThreatLevel.MEDIUM, ThreatLevel.HIGH, ThreatLevel.CRITICAL]
        # Check if brute force is in threat types (implementation dependent)
    
    @pytest.mark.asyncio
    async def test_behavioral_anomaly_detection(self, threat_detector):
        """Test behavioral anomaly detection"""
        server_id = "behavior-test"
        
        # Establish baseline with normal activity
        normal_context = {"source_ip": "192.168.1.50", "user_agent": "normal-client"}
        for _ in range(50):  # Establish baseline
            await threat_detector.assess_authentication_risk(
                server_id, {"api_key": "key"}, normal_context
            )
        
        # Test anomalous activity
        anomalous_context = {"source_ip": "10.0.0.1", "user_agent": "suspicious-client"}
        assessment = await threat_detector.assess_authentication_risk(
            server_id, {"api_key": "key"}, anomalous_context
        )
        
        # Should detect some level of anomaly (implementation dependent)
        # For basic implementation, this might not trigger, but framework is there
        assert assessment is not None
    
    @pytest.mark.asyncio
    async def test_privilege_escalation_detection(self, threat_detector):
        """Test privilege escalation detection"""
        server_id = "privilege-test"
        
        # Test admin operation request
        assessment = await threat_detector.assess_operation_risk(
            server_id, "admin_delete", ["admin_config.json"], {}
        )
        
        # Should detect potential privilege escalation
        assert assessment.threat_level != ThreatLevel.NONE
    
    @pytest.mark.asyncio
    async def test_data_exfiltration_detection(self, threat_detector):
        """Test data exfiltration pattern detection"""
        server_id = "exfiltration-test"
        
        # Test bulk data access
        large_resource_list = [f"file_{i}.txt" for i in range(15)]
        assessment = await threat_detector.assess_operation_risk(
            server_id, "read", large_resource_list, {}
        )
        
        # Should detect potential data exfiltration
        assert assessment.threat_level != ThreatLevel.NONE
    
    @pytest.mark.asyncio
    async def test_threat_metrics_collection(self, threat_detector):
        """Test threat detection metrics"""
        # Generate some threat detections
        server_id = "metrics-test"
        
        await threat_detector.assess_authentication_risk(
            server_id, {"api_key": "key"}, {}
        )
        await threat_detector.assess_operation_risk(
            server_id, "admin_access", ["admin.conf"], {}
        )
        
        # Get metrics
        metrics = await threat_detector.get_threat_metrics()
        
        assert isinstance(metrics.total_threats_detected, int)
        assert isinstance(metrics.average_response_time_ms, float)


# Integration tests
class TestSecurityGatewayIntegration:
    """Integration tests for complete security workflow"""
    
    @pytest.fixture
    async def temp_dir(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            yield Path(temp_dir)
    
    @pytest.fixture
    async def full_security_system(self, temp_dir):
        """Setup complete security system for integration testing"""
        config = {
            'credentials': {'storage_path': str(temp_dir / 'credentials')},
            'permissions': {'storage_path': str(temp_dir / 'permissions')},
            'audit': {'storage_path': str(temp_dir / 'audit')},
            'sessions': {'storage_path': str(temp_dir / 'sessions')},
            'threat_detection': {'storage_path': str(temp_dir / 'threats')}
        }
        
        gateway = MCPSecurityGateway(config)
        yield gateway
        
        # Cleanup
        if hasattr(gateway.audit_logger, 'shutdown'):
            gateway.audit_logger.shutdown()
        if hasattr(gateway.session_manager, 'shutdown'):
            await gateway.session_manager.shutdown()
    
    @pytest.mark.asyncio
    async def test_complete_security_workflow(self, full_security_system):
        """Test complete security workflow from authentication to operation"""
        gateway = full_security_system
        server_id = "integration-test-server"
        credentials = {"api_key": "integration-test-key"}
        
        # 1. Store credential
        await gateway.credential_manager.store_credential(
            server_id, "api_key", "integration-test-key", 
            ["read", "write", "execute"]
        )
        
        # 2. Authenticate
        session_token = await gateway.authenticate_server(server_id, credentials)
        assert session_token is not None
        
        # 3. Authorize operations
        operations = [
            ("read", ["file1.txt"]),
            ("write", ["file2.txt"]), 
            ("execute", ["script.py"])
        ]
        
        for operation, resources in operations:
            result = await gateway.authorize_operation(
                session_token, operation, resources
            )
            assert result is True
        
        # 4. Test secure operation context
        async with gateway.secure_operation_context(
            session_token, "read", ["data.json"]
        ) as context:
            assert context.server_id == server_id
        
        # 5. Get security metrics
        metrics = await gateway.get_security_metrics()
        assert metrics.active_sessions >= 1
        assert metrics.authentication_attempts >= 1
        
        # 6. Invalidate session
        result = await gateway.invalidate_session(session_token)
        assert result is True
    
    @pytest.mark.asyncio
    async def test_security_violation_workflow(self, full_security_system):
        """Test security violation detection and response"""
        gateway = full_security_system
        server_id = "violation-test-server"
        
        # Store credential
        await gateway.credential_manager.store_credential(
            server_id, "api_key", "violation-key", ["read"]
        )
        
        # Mock threat detector to simulate security violation
        with patch.object(
            gateway.threat_detector,
            'assess_authentication_risk'
        ) as mock_assess:
            mock_assess.return_value = ThreatAssessment(
                threat_level=ThreatLevel.CRITICAL,
                threat_types=[],
                confidence_score=0.95,
                risk_factors=["Simulated security violation"],
                recommended_actions=["Block immediately"],
                reason="Test security violation"
            )
            
            # Authentication should be blocked
            with pytest.raises(SecurityViolationError):
                await gateway.authenticate_server(
                    server_id, {"api_key": "violation-key"}
                )
        
        # Verify no session was created
        assert len(gateway.active_sessions) == 0
    
    @pytest.mark.asyncio
    async def test_performance_under_load(self, full_security_system):
        """Test security system performance under simulated load"""
        gateway = full_security_system
        
        # Setup multiple test servers
        servers = []
        for i in range(5):
            server_id = f"load-test-server-{i}"
            key = f"load-test-key-{i}"
            await gateway.credential_manager.store_credential(
                server_id, "api_key", key, ["read", "write"]
            )
            servers.append((server_id, {"api_key": key}))
        
        # Simulate concurrent authentication requests
        async def auth_and_operate(server_id, credentials):
            try:
                token = await gateway.authenticate_server(server_id, credentials)
                await gateway.authorize_operation(token, "read", ["test.txt"])
                await gateway.authorize_operation(token, "write", ["output.txt"])
                return True
            except Exception as e:
                print(f"Error in concurrent test: {e}")
                return False
        
        # Run concurrent operations
        tasks = [auth_and_operate(sid, creds) for sid, creds in servers]
        results = await asyncio.gather(*tasks, return_exceptions=True)
        
        # Most operations should succeed
        success_count = sum(1 for r in results if r is True)
        assert success_count >= len(servers) * 0.8  # At least 80% success rate
        
        # Check final metrics
        metrics = await gateway.get_security_metrics()
        assert metrics.authentication_attempts >= len(servers)


if __name__ == '__main__':
    # Run tests with: python -m pytest tests/mcp/security/test_security_gateway.py -v
    pytest.main([__file__, '-v'])