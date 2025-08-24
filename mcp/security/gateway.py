"""
MCP Security Gateway - Core Security Controller

Implements enterprise-grade security gateway for MCP server integration with:
- Zero-trust security model
- Multi-factor authentication
- RBAC authorization with least privilege
- Real-time threat detection
- Comprehensive security monitoring
"""

import asyncio
import logging
import time
from typing import Dict, List, Optional, Any, Tuple
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from contextlib import asynccontextmanager

from .credential_manager import CredentialManager
from .permission_matrix import PermissionMatrix
from .audit_logger import AuditLogger
from .session_manager import SessionManager
from .threat_detector import ThreatDetector
from .exceptions import (
    SecurityGatewayError,
    AuthenticationError,
    AuthorizationError,
    SecurityViolationError
)


@dataclass
class SecurityContext:
    """Security context for authenticated sessions"""
    server_id: str
    session_token: str
    authenticated_at: datetime
    permissions: List[str] = field(default_factory=list)
    resources_access: Dict[str, List[str]] = field(default_factory=dict)
    last_activity: datetime = field(default_factory=datetime.utcnow)
    threat_level: str = "low"
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass 
class SecurityMetrics:
    """Real-time security metrics"""
    active_sessions: int = 0
    authentication_attempts: int = 0
    authentication_failures: int = 0
    authorization_failures: int = 0
    security_violations: int = 0
    threat_detections: int = 0
    average_auth_latency: float = 0.0
    average_authz_latency: float = 0.0
    
    
class MCPSecurityGateway:
    """
    Enterprise-grade MCP Security Gateway implementing zero-trust security model.
    
    Provides centralized security enforcement for all MCP server interactions with:
    - Multi-factor authentication with automatic credential rotation
    - RBAC-based authorization with least privilege enforcement  
    - Real-time threat detection and anomaly monitoring
    - Comprehensive audit logging and security event tracking
    - Session management with secure token generation
    - Performance-optimized security operations (<200ms auth, <100ms authz)
    """
    
    def __init__(self, config: Optional[Dict[str, Any]] = None):
        """
        Initialize MCP Security Gateway with enterprise security configuration.
        
        Args:
            config: Optional security configuration overrides
        """
        self.config = self._load_security_config(config or {})
        self.logger = logging.getLogger(__name__)
        
        # Initialize security components
        self.credential_manager = CredentialManager(self.config.get('credentials', {}))
        self.permission_matrix = PermissionMatrix(self.config.get('permissions', {}))
        self.audit_logger = AuditLogger(self.config.get('audit', {}))
        self.session_manager = SessionManager(self.config.get('sessions', {}))
        self.threat_detector = ThreatDetector(self.config.get('threat_detection', {}))
        
        # Security state tracking
        self.active_sessions: Dict[str, SecurityContext] = {}
        self.metrics = SecurityMetrics()
        self._security_lock = asyncio.Lock()
        
        # Performance monitoring
        self._auth_latencies: List[float] = []
        self._authz_latencies: List[float] = []
        
        self.logger.info("MCP Security Gateway initialized with zero-trust security model")
    
    async def authenticate_server(
        self, 
        server_id: str, 
        credentials: Dict[str, Any],
        context: Optional[Dict[str, Any]] = None
    ) -> str:
        """
        Authenticate MCP server with multi-factor validation.
        
        Implements zero-trust authentication with:
        - Multi-factor credential validation
        - Real-time threat detection
        - Performance monitoring (<200ms target)
        - Comprehensive security logging
        
        Args:
            server_id: Unique server identifier
            credentials: Authentication credentials
            context: Optional authentication context
            
        Returns:
            Secure session token for authenticated session
            
        Raises:
            AuthenticationError: Authentication failed
            SecurityViolationError: Security policy violation detected
        """
        start_time = time.perf_counter()
        context = context or {}
        
        try:
            async with self._security_lock:
                # Increment authentication attempt metrics
                self.metrics.authentication_attempts += 1
                
                # Check for active threats or security violations
                threat_assessment = await self.threat_detector.assess_authentication_risk(
                    server_id, credentials, context
                )
                
                if threat_assessment.threat_level == "critical":
                    await self.audit_logger.log_security_violation(
                        server_id, "authentication_blocked", threat_assessment
                    )
                    raise SecurityViolationError(f"Authentication blocked due to security threat: {threat_assessment.reason}")
                
                # Validate credentials through multiple factors
                auth_result = await self.credential_manager.validate_credentials(
                    server_id, credentials
                )
                
                if not auth_result.is_valid:
                    self.metrics.authentication_failures += 1
                    await self.audit_logger.log_failed_authentication(
                        server_id, auth_result.failure_reason, context
                    )
                    raise AuthenticationError(f"Authentication failed: {auth_result.failure_reason}")
                
                # Generate secure session token
                session_token = await self.session_manager.generate_session_token(
                    server_id, auth_result.permissions
                )
                
                # Create security context
                security_context = SecurityContext(
                    server_id=server_id,
                    session_token=session_token,
                    authenticated_at=datetime.utcnow(),
                    permissions=auth_result.permissions,
                    resources_access=auth_result.resource_access,
                    threat_level=threat_assessment.threat_level,
                    metadata={
                        **context,
                        'auth_method': auth_result.auth_method,
                        'credential_rotation_due': auth_result.rotation_due
                    }
                )
                
                # Store active session
                self.active_sessions[session_token] = security_context
                self.metrics.active_sessions += 1
                
                # Log successful authentication
                await self.audit_logger.log_successful_authentication(
                    server_id, session_token, {
                        "auth_method": auth_result.auth_method,
                        "permissions": security_context.permissions,
                        "threat_level": security_context.threat_level,
                        "authenticated_at": security_context.authenticated_at.isoformat()
                    }
                )
                
                # Performance tracking
                auth_latency = (time.perf_counter() - start_time) * 1000
                self._auth_latencies.append(auth_latency)
                if len(self._auth_latencies) > 100:
                    self._auth_latencies.pop(0)
                self.metrics.average_auth_latency = sum(self._auth_latencies) / len(self._auth_latencies)
                
                # Check if authentication exceeded performance target
                if auth_latency > 200:
                    self.logger.warning(f"Authentication latency exceeded target: {auth_latency:.2f}ms")
                
                self.logger.info(f"Server {server_id} authenticated successfully in {auth_latency:.2f}ms")
                return session_token
                
        except Exception as e:
            self.metrics.authentication_failures += 1
            await self.audit_logger.log_authentication_error(server_id, str(e), context)
            raise
    
    async def authorize_operation(
        self,
        session_token: str,
        operation: str,
        resources: List[str],
        context: Optional[Dict[str, Any]] = None
    ) -> bool:
        """
        Authorize operation with RBAC and least privilege enforcement.
        
        Implements comprehensive authorization with:
        - RBAC-based permission validation
        - Least privilege enforcement
        - Resource-level access control
        - Real-time threat monitoring
        - Performance optimization (<100ms target)
        
        Args:
            session_token: Valid session token from authentication
            operation: Operation being requested
            resources: List of resources being accessed
            context: Optional operation context
            
        Returns:
            True if operation is authorized
            
        Raises:
            AuthorizationError: Operation not authorized
            SecurityViolationError: Security policy violation
        """
        start_time = time.perf_counter()
        context = context or {}
        
        try:
            # Validate session token
            security_context = await self._validate_session_token(session_token)
            
            # Check for security threats
            threat_assessment = await self.threat_detector.assess_operation_risk(
                security_context.server_id, operation, resources, context
            )
            
            if threat_assessment.threat_level in ["high", "critical"]:
                await self.audit_logger.log_security_violation(
                    security_context.server_id, "operation_blocked", threat_assessment
                )
                raise SecurityViolationError(f"Operation blocked due to security threat: {threat_assessment.reason}")
            
            # RBAC permission validation with least privilege
            permission_result = await self.permission_matrix.check_permissions(
                security_context.permissions,
                operation,
                resources,
                security_context.resources_access
            )
            
            if not permission_result.is_allowed:
                self.metrics.authorization_failures += 1
                await self.audit_logger.log_authorization_failure(
                    security_context.server_id, operation, resources, permission_result.denial_reason
                )
                raise AuthorizationError(f"Operation not authorized: {permission_result.denial_reason}")
            
            # Update session activity
            security_context.last_activity = datetime.utcnow()
            
            # Log authorized operation
            await self.audit_logger.log_authorized_operation(
                security_context.server_id, operation, resources, permission_result
            )
            
            # Performance tracking
            authz_latency = (time.perf_counter() - start_time) * 1000
            self._authz_latencies.append(authz_latency)
            if len(self._authz_latencies) > 100:
                self._authz_latencies.pop(0)
            self.metrics.average_authz_latency = sum(self._authz_latencies) / len(self._authz_latencies)
            
            # Check if authorization exceeded performance target
            if authz_latency > 100:
                self.logger.warning(f"Authorization latency exceeded target: {authz_latency:.2f}ms")
            
            return True
            
        except Exception as e:
            self.metrics.authorization_failures += 1
            await self.audit_logger.log_authorization_error(session_token, operation, str(e))
            raise
    
    async def rotate_credentials(
        self, 
        server_id: Optional[str] = None,
        force_rotation: bool = False
    ) -> Dict[str, str]:
        """
        Rotate credentials with zero-downtime for specified server or all servers.
        
        Args:
            server_id: Optional specific server, None for all servers
            force_rotation: Force rotation regardless of schedule
            
        Returns:
            Dictionary of server_id -> new_credential_info
        """
        self.logger.info(f"Starting credential rotation for server: {server_id or 'all'}")
        
        try:
            if server_id:
                # Rotate specific server credentials
                new_credentials = await self.credential_manager.rotate_server_credentials(
                    server_id, force_rotation
                )
                
                # Notify server of new credentials
                await self._notify_server_credential_rotation(server_id, new_credentials)
                
                await self.audit_logger.log_credential_rotation(server_id, "success")
                return {server_id: new_credentials}
            else:
                # Rotate all server credentials
                rotation_results = {}
                
                for sid in await self.credential_manager.get_all_server_ids():
                    try:
                        new_creds = await self.credential_manager.rotate_server_credentials(
                            sid, force_rotation
                        )
                        await self._notify_server_credential_rotation(sid, new_creds)
                        rotation_results[sid] = new_creds
                        
                        await self.audit_logger.log_credential_rotation(sid, "success")
                        
                    except Exception as e:
                        self.logger.error(f"Failed to rotate credentials for server {sid}: {e}")
                        await self.audit_logger.log_credential_rotation(sid, "failure", str(e))
                        rotation_results[sid] = f"rotation_failed: {str(e)}"
                
                return rotation_results
                
        except Exception as e:
            await self.audit_logger.log_credential_rotation(
                server_id or "all", "error", str(e)
            )
            raise SecurityGatewayError(f"Credential rotation failed: {e}")
    
    async def invalidate_session(self, session_token: str, reason: str = "manual") -> bool:
        """
        Invalidate session token and cleanup security context.
        
        Args:
            session_token: Session token to invalidate
            reason: Reason for invalidation
            
        Returns:
            True if session was successfully invalidated
        """
        try:
            if session_token in self.active_sessions:
                security_context = self.active_sessions[session_token]
                
                # Invalidate in session manager
                await self.session_manager.invalidate_session(session_token)
                
                # Remove from active sessions
                del self.active_sessions[session_token]
                self.metrics.active_sessions -= 1
                
                # Log session invalidation
                await self.audit_logger.log_session_invalidated(
                    security_context.server_id, session_token, reason
                )
                
                self.logger.info(f"Session invalidated for server {security_context.server_id}: {reason}")
                return True
            
            return False
            
        except Exception as e:
            self.logger.error(f"Failed to invalidate session {session_token}: {e}")
            return False
    
    async def get_security_metrics(self) -> SecurityMetrics:
        """Get real-time security metrics for monitoring"""
        # Update threat detection metrics
        threat_metrics = await self.threat_detector.get_threat_metrics()
        self.metrics.threat_detections = threat_metrics.total_threats_detected
        
        return self.metrics
    
    @asynccontextmanager
    async def secure_operation_context(
        self,
        session_token: str,
        operation: str,
        resources: List[str]
    ):
        """
        Context manager for secure operations with automatic cleanup.
        
        Usage:
            async with gateway.secure_operation_context(token, "read", ["file1"]) as ctx:
                # Perform secure operation
                pass
        """
        # Authorize operation
        await self.authorize_operation(session_token, operation, resources)
        
        try:
            yield self.active_sessions.get(session_token)
        finally:
            # Log operation completion
            if session_token in self.active_sessions:
                await self.audit_logger.log_operation_completed(
                    self.active_sessions[session_token].server_id,
                    operation,
                    resources
                )
    
    async def cleanup_expired_sessions(self) -> int:
        """
        Cleanup expired sessions based on inactivity timeout.
        
        Returns:
            Number of sessions cleaned up
        """
        current_time = datetime.utcnow()
        expired_tokens = []
        session_timeout = timedelta(hours=self.config.get('session_timeout_hours', 24))
        
        for token, context in self.active_sessions.items():
            if current_time - context.last_activity > session_timeout:
                expired_tokens.append(token)
        
        cleanup_count = 0
        for token in expired_tokens:
            if await self.invalidate_session(token, "expired"):
                cleanup_count += 1
        
        if cleanup_count > 0:
            self.logger.info(f"Cleaned up {cleanup_count} expired sessions")
        
        return cleanup_count
    
    async def _validate_session_token(self, session_token: str) -> SecurityContext:
        """Validate session token and return security context"""
        if not session_token or session_token not in self.active_sessions:
            raise AuthenticationError("Invalid or expired session token")
        
        security_context = self.active_sessions[session_token]
        
        # Check session validity with session manager
        is_valid = await self.session_manager.validate_session_token(session_token)
        if not is_valid:
            # Cleanup invalid session
            await self.invalidate_session(session_token, "invalid")
            raise AuthenticationError("Session token validation failed")
        
        return security_context
    
    async def _notify_server_credential_rotation(
        self, 
        server_id: str, 
        new_credentials: str
    ) -> None:
        """Notify server of credential rotation (implementation depends on server communication)"""
        # This would integrate with the server communication mechanism
        # For now, log the rotation notification
        self.logger.info(f"Credential rotation notification sent to server {server_id}")
        
    def _load_security_config(self, config: Dict[str, Any]) -> Dict[str, Any]:
        """Load and validate security configuration"""
        default_config = {
            'session_timeout_hours': 24,
            'auth_performance_target_ms': 200,
            'authz_performance_target_ms': 100,
            'threat_detection_enabled': True,
            'audit_retention_days': 90,
            'credential_rotation_interval_hours': 168,  # 7 days
            'max_concurrent_sessions_per_server': 10
        }
        
        return {**default_config, **config}