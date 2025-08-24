"""
MCP Session Manager - Secure Session Token Management

Implements enterprise-grade session management with:
- Secure token generation with cryptographic randomness
- JWT-based session tokens with claims and expiration
- Session lifecycle management with automatic cleanup
- Token validation with integrity verification
- Session hijacking prevention with security fingerprinting
"""

import asyncio
import hashlib
import hmac
import json
import secrets
import time
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Any, NamedTuple
from dataclasses import dataclass, field
from pathlib import Path
import jwt
import logging
from cryptography.hazmat.primitives import serialization
from cryptography.hazmat.primitives.asymmetric import rsa


class TokenValidationResult(NamedTuple):
    """Result of token validation"""
    is_valid: bool
    server_id: Optional[str] = None
    permissions: List[str] = []
    expires_at: Optional[datetime] = None
    error_reason: Optional[str] = None


@dataclass
class SessionInfo:
    """Session information stored in manager"""
    session_token: str
    server_id: str
    created_at: datetime
    last_activity: datetime
    expires_at: datetime
    permissions: List[str]
    security_fingerprint: str
    is_active: bool = True
    metadata: Dict[str, Any] = field(default_factory=dict)
    
    @property
    def is_expired(self) -> bool:
        """Check if session has expired"""
        return datetime.utcnow() > self.expires_at
    
    @property
    def time_until_expiry(self) -> timedelta:
        """Time remaining until session expires"""
        return max(timedelta(0), self.expires_at - datetime.utcnow())


@dataclass
class SessionMetrics:
    """Session management metrics"""
    active_sessions: int = 0
    total_sessions_created: int = 0
    total_sessions_expired: int = 0
    total_sessions_invalidated: int = 0
    average_session_duration_minutes: float = 0.0
    security_violations: int = 0
    token_validation_failures: int = 0


class SessionManager:
    """
    Enterprise-grade session manager for MCP security gateway.
    
    Features:
    - JWT-based secure token generation with RSA signatures
    - Session lifecycle management with automatic cleanup
    - Security fingerprinting to prevent session hijacking
    - Token validation with cryptographic integrity checks
    - Configurable session timeout and cleanup policies
    """
    
    def __init__(self, config: Dict[str, Any]):
        """
        Initialize session manager with security configuration.
        
        Args:
            config: Session management configuration
        """
        self.config = config
        self.logger = logging.getLogger(__name__)
        
        # Initialize storage
        self.storage_path = Path(config.get('storage_path', 'knowledge/security/sessions'))
        self.storage_path.mkdir(parents=True, exist_ok=True)
        
        # Session configuration
        self.default_session_timeout = timedelta(
            hours=config.get('default_session_timeout_hours', 24)
        )
        self.cleanup_interval = timedelta(
            minutes=config.get('cleanup_interval_minutes', 30)
        )
        self.max_sessions_per_server = config.get('max_sessions_per_server', 10)
        
        # JWT configuration
        self.jwt_algorithm = config.get('jwt_algorithm', 'RS256')
        self.jwt_issuer = config.get('jwt_issuer', 'mcp-security-gateway')
        
        # Initialize RSA keys for JWT signing
        self._private_key_path = self.storage_path / '.session_private.pem'
        self._public_key_path = self.storage_path / '.session_public.pem'
        self._private_key, self._public_key = self._load_or_create_rsa_keys()
        
        # Session storage
        self.active_sessions: Dict[str, SessionInfo] = {}
        self._session_lock = asyncio.Lock()
        
        # Metrics
        self.metrics = SessionMetrics()
        
        # Start background cleanup
        self._cleanup_task: Optional[asyncio.Task] = None
        self._start_cleanup_task()
        
        self.logger.info("Session Manager initialized with secure JWT token generation")
    
    async def generate_session_token(
        self,
        server_id: str,
        permissions: List[str],
        session_timeout: Optional[timedelta] = None,
        security_context: Optional[Dict[str, Any]] = None
    ) -> str:
        """
        Generate secure JWT session token for authenticated server.
        
        Args:
            server_id: Server identifier
            permissions: List of permissions for this session
            session_timeout: Optional custom session timeout
            security_context: Optional security context for fingerprinting
            
        Returns:
            JWT session token
            
        Raises:
            ValueError: If session creation fails
        """
        try:
            async with self._session_lock:
                # Check session limits per server
                existing_sessions = [
                    s for s in self.active_sessions.values()
                    if s.server_id == server_id and s.is_active and not s.is_expired
                ]
                
                if len(existing_sessions) >= self.max_sessions_per_server:
                    # Cleanup oldest session
                    oldest_session = min(existing_sessions, key=lambda s: s.last_activity)
                    await self._invalidate_session_internal(oldest_session.session_token, "session_limit_exceeded")
                
                # Generate session details
                now = datetime.utcnow()
                timeout = session_timeout or self.default_session_timeout
                expires_at = now + timeout
                
                # Create security fingerprint
                security_fingerprint = self._create_security_fingerprint(
                    server_id, security_context or {}
                )
                
                # Generate unique session ID
                session_id = self._generate_session_id()
                
                # Create JWT claims
                jwt_claims = {
                    'iss': self.jwt_issuer,  # Issuer
                    'sub': server_id,  # Subject (server_id)
                    'iat': int(now.timestamp()),  # Issued at
                    'exp': int(expires_at.timestamp()),  # Expires at
                    'jti': session_id,  # JWT ID (unique token identifier)
                    'permissions': permissions,
                    'security_fingerprint': security_fingerprint,
                    'session_metadata': {
                        'created_at': now.isoformat(),
                        'timeout_seconds': int(timeout.total_seconds()),
                        'max_inactivity_minutes': self.config.get('max_inactivity_minutes', 60)
                    }
                }
                
                # Generate JWT token
                session_token = jwt.encode(
                    jwt_claims,
                    self._private_key,
                    algorithm=self.jwt_algorithm
                )
                
                # Store session info
                session_info = SessionInfo(
                    session_token=session_token,
                    server_id=server_id,
                    created_at=now,
                    last_activity=now,
                    expires_at=expires_at,
                    permissions=permissions,
                    security_fingerprint=security_fingerprint,
                    metadata=security_context or {}
                )
                
                self.active_sessions[session_token] = session_info
                
                # Update metrics
                self.metrics.total_sessions_created += 1
                self.metrics.active_sessions = len([s for s in self.active_sessions.values() if s.is_active])
                
                self.logger.info(f"Session token generated for server {server_id}, expires at {expires_at}")
                return session_token
                
        except Exception as e:
            self.logger.error(f"Failed to generate session token for server {server_id}: {e}")
            raise ValueError(f"Session token generation failed: {e}")
    
    async def validate_session_token(
        self,
        session_token: str,
        security_context: Optional[Dict[str, Any]] = None
    ) -> TokenValidationResult:
        """
        Validate JWT session token and return validation result.
        
        Args:
            session_token: JWT session token to validate
            security_context: Optional security context for fingerprint validation
            
        Returns:
            TokenValidationResult with validation status and details
        """
        try:
            # Decode and verify JWT
            try:
                decoded_token = jwt.decode(
                    session_token,
                    self._public_key,
                    algorithms=[self.jwt_algorithm],
                    issuer=self.jwt_issuer
                )
            except jwt.ExpiredSignatureError:
                self.metrics.token_validation_failures += 1
                return TokenValidationResult(
                    is_valid=False,
                    error_reason="Token has expired"
                )
            except jwt.InvalidTokenError as e:
                self.metrics.token_validation_failures += 1
                return TokenValidationResult(
                    is_valid=False,
                    error_reason=f"Invalid token: {str(e)}"
                )
            
            # Check if session exists in active sessions
            if session_token not in self.active_sessions:
                self.metrics.token_validation_failures += 1
                return TokenValidationResult(
                    is_valid=False,
                    error_reason="Session not found in active sessions"
                )
            
            session_info = self.active_sessions[session_token]
            
            # Check if session is active and not expired
            if not session_info.is_active or session_info.is_expired:
                self.metrics.token_validation_failures += 1
                return TokenValidationResult(
                    is_valid=False,
                    error_reason="Session is inactive or expired"
                )
            
            # Validate security fingerprint if provided
            if security_context:
                expected_fingerprint = self._create_security_fingerprint(
                    session_info.server_id, security_context
                )
                if expected_fingerprint != session_info.security_fingerprint:
                    self.metrics.security_violations += 1
                    await self._handle_security_fingerprint_mismatch(session_token, session_info)
                    return TokenValidationResult(
                        is_valid=False,
                        error_reason="Security fingerprint mismatch - possible session hijacking"
                    )
            
            # Update last activity
            session_info.last_activity = datetime.utcnow()
            
            # Return successful validation
            return TokenValidationResult(
                is_valid=True,
                server_id=decoded_token['sub'],
                permissions=decoded_token['permissions'],
                expires_at=datetime.fromtimestamp(decoded_token['exp'])
            )
            
        except Exception as e:
            self.logger.error(f"Session validation error for token: {e}")
            self.metrics.token_validation_failures += 1
            return TokenValidationResult(
                is_valid=False,
                error_reason=f"Validation error: {str(e)}"
            )
    
    async def invalidate_session(
        self,
        session_token: str,
        reason: str = "manual"
    ) -> bool:
        """
        Invalidate session token.
        
        Args:
            session_token: Session token to invalidate
            reason: Reason for invalidation
            
        Returns:
            True if session was successfully invalidated
        """
        return await self._invalidate_session_internal(session_token, reason)
    
    async def invalidate_all_sessions_for_server(
        self,
        server_id: str,
        reason: str = "server_logout"
    ) -> int:
        """
        Invalidate all sessions for a specific server.
        
        Args:
            server_id: Server to invalidate sessions for
            reason: Reason for invalidation
            
        Returns:
            Number of sessions invalidated
        """
        try:
            async with self._session_lock:
                sessions_to_invalidate = [
                    token for token, session in self.active_sessions.items()
                    if session.server_id == server_id and session.is_active
                ]
                
                invalidated_count = 0
                for token in sessions_to_invalidate:
                    if await self._invalidate_session_internal(token, reason):
                        invalidated_count += 1
                
                self.logger.info(f"Invalidated {invalidated_count} sessions for server {server_id}")
                return invalidated_count
                
        except Exception as e:
            self.logger.error(f"Failed to invalidate sessions for server {server_id}: {e}")
            return 0
    
    async def refresh_session_token(
        self,
        current_token: str,
        extend_timeout: Optional[timedelta] = None
    ) -> Optional[str]:
        """
        Refresh session token with new expiration time.
        
        Args:
            current_token: Current valid session token
            extend_timeout: Optional timeout extension
            
        Returns:
            New session token or None if refresh failed
        """
        try:
            # Validate current token first
            validation_result = await self.validate_session_token(current_token)
            if not validation_result.is_valid:
                return None
            
            async with self._session_lock:
                if current_token not in self.active_sessions:
                    return None
                
                current_session = self.active_sessions[current_token]
                
                # Generate new token with same permissions but extended timeout
                new_timeout = extend_timeout or self.default_session_timeout
                new_token = await self.generate_session_token(
                    current_session.server_id,
                    current_session.permissions,
                    new_timeout,
                    current_session.metadata
                )
                
                # Invalidate old token
                await self._invalidate_session_internal(current_token, "token_refreshed")
                
                self.logger.info(f"Session token refreshed for server {current_session.server_id}")
                return new_token
                
        except Exception as e:
            self.logger.error(f"Failed to refresh session token: {e}")
            return None
    
    async def get_session_info(self, session_token: str) -> Optional[Dict[str, Any]]:
        """
        Get session information for a valid token.
        
        Args:
            session_token: Session token to get info for
            
        Returns:
            Session information dictionary or None if not found
        """
        try:
            if session_token not in self.active_sessions:
                return None
            
            session_info = self.active_sessions[session_token]
            
            return {
                "server_id": session_info.server_id,
                "created_at": session_info.created_at.isoformat(),
                "last_activity": session_info.last_activity.isoformat(),
                "expires_at": session_info.expires_at.isoformat(),
                "permissions": session_info.permissions,
                "is_active": session_info.is_active,
                "time_until_expiry_minutes": session_info.time_until_expiry.total_seconds() / 60,
                "metadata": session_info.metadata
            }
            
        except Exception as e:
            self.logger.error(f"Failed to get session info: {e}")
            return None
    
    async def cleanup_expired_sessions(self) -> int:
        """
        Clean up expired and inactive sessions.
        
        Returns:
            Number of sessions cleaned up
        """
        try:
            async with self._session_lock:
                expired_tokens = []
                inactive_timeout = timedelta(minutes=self.config.get('max_inactivity_minutes', 60))
                current_time = datetime.utcnow()
                
                for token, session in self.active_sessions.items():
                    if (session.is_expired or 
                        not session.is_active or
                        current_time - session.last_activity > inactive_timeout):
                        expired_tokens.append(token)
                
                cleanup_count = 0
                for token in expired_tokens:
                    if await self._invalidate_session_internal(token, "expired"):
                        cleanup_count += 1
                
                if cleanup_count > 0:
                    self.logger.info(f"Cleaned up {cleanup_count} expired sessions")
                
                return cleanup_count
                
        except Exception as e:
            self.logger.error(f"Failed to cleanup expired sessions: {e}")
            return 0
    
    async def get_session_metrics(self) -> SessionMetrics:
        """Get current session metrics"""
        # Update active session count
        self.metrics.active_sessions = len([
            s for s in self.active_sessions.values()
            if s.is_active and not s.is_expired
        ])
        
        return self.metrics
    
    def _generate_session_id(self) -> str:
        """Generate unique session identifier"""
        return f"sess_{int(datetime.utcnow().timestamp() * 1000000)}_{secrets.token_hex(16)}"
    
    def _create_security_fingerprint(
        self,
        server_id: str,
        security_context: Dict[str, Any]
    ) -> str:
        """
        Create security fingerprint to prevent session hijacking.
        
        Args:
            server_id: Server identifier
            security_context: Security context with client information
            
        Returns:
            Security fingerprint hash
        """
        # Create fingerprint from server ID and security context
        fingerprint_data = [
            server_id,
            security_context.get('client_ip', ''),
            security_context.get('user_agent', ''),
            security_context.get('client_certificate_hash', ''),
            str(security_context.get('client_capabilities', []))
        ]
        
        fingerprint_string = "|".join(fingerprint_data)
        return hashlib.sha256(fingerprint_string.encode('utf-8')).hexdigest()
    
    async def _invalidate_session_internal(
        self,
        session_token: str,
        reason: str
    ) -> bool:
        """Internal session invalidation"""
        try:
            async with self._session_lock:
                if session_token not in self.active_sessions:
                    return False
                
                session_info = self.active_sessions[session_token]
                session_info.is_active = False
                
                # Calculate session duration for metrics
                session_duration = datetime.utcnow() - session_info.created_at
                
                # Update metrics
                self.metrics.total_sessions_invalidated += 1
                if reason == "expired":
                    self.metrics.total_sessions_expired += 1
                
                # Update average session duration
                if self.metrics.total_sessions_invalidated > 0:
                    current_total_minutes = (
                        self.metrics.average_session_duration_minutes * 
                        (self.metrics.total_sessions_invalidated - 1)
                    )
                    new_total_minutes = current_total_minutes + (session_duration.total_seconds() / 60)
                    self.metrics.average_session_duration_minutes = (
                        new_total_minutes / self.metrics.total_sessions_invalidated
                    )
                
                # Remove from active sessions after a delay (for audit purposes)
                asyncio.create_task(self._delayed_session_removal(session_token))
                
                self.logger.info(f"Session invalidated for server {session_info.server_id}: {reason}")
                return True
                
        except Exception as e:
            self.logger.error(f"Failed to invalidate session {session_token}: {e}")
            return False
    
    async def _delayed_session_removal(self, session_token: str, delay_minutes: int = 5) -> None:
        """Remove session from memory after delay for audit purposes"""
        try:
            await asyncio.sleep(delay_minutes * 60)
            
            async with self._session_lock:
                if session_token in self.active_sessions:
                    del self.active_sessions[session_token]
                    
        except Exception as e:
            self.logger.error(f"Failed to remove session {session_token}: {e}")
    
    async def _handle_security_fingerprint_mismatch(
        self,
        session_token: str,
        session_info: SessionInfo
    ) -> None:
        """Handle security fingerprint mismatch (possible session hijacking)"""
        try:
            # Immediately invalidate session
            await self._invalidate_session_internal(session_token, "security_violation")
            
            # Log security incident
            self.logger.critical(
                f"Security fingerprint mismatch for server {session_info.server_id} "
                f"- possible session hijacking detected"
            )
            
            # This would trigger security alerts in production
            
        except Exception as e:
            self.logger.error(f"Failed to handle security fingerprint mismatch: {e}")
    
    def _load_or_create_rsa_keys(self) -> tuple:
        """Load existing RSA keys or create new ones for JWT signing"""
        try:
            # Try to load existing keys
            if self._private_key_path.exists() and self._public_key_path.exists():
                with open(self._private_key_path, 'rb') as f:
                    private_key = serialization.load_pem_private_key(f.read(), password=None)
                
                with open(self._public_key_path, 'rb') as f:
                    public_key = serialization.load_pem_public_key(f.read())
                
                self.logger.info("Loaded existing RSA keys for JWT signing")
                return private_key, public_key
                
        except Exception as e:
            self.logger.warning(f"Failed to load existing RSA keys: {e}, creating new ones")
        
        # Create new RSA key pair
        try:
            private_key = rsa.generate_private_key(
                public_exponent=65537,
                key_size=2048
            )
            public_key = private_key.public_key()
            
            # Save private key
            with open(self._private_key_path, 'wb') as f:
                f.write(private_key.private_bytes(
                    encoding=serialization.Encoding.PEM,
                    format=serialization.PrivateFormat.PKCS8,
                    encryption_algorithm=serialization.NoEncryption()
                ))
            
            # Save public key
            with open(self._public_key_path, 'wb') as f:
                f.write(public_key.public_bytes(
                    encoding=serialization.Encoding.PEM,
                    format=serialization.PublicFormat.SubjectPublicKeyInfo
                ))
            
            # Set secure file permissions
            self._private_key_path.chmod(0o600)
            self._public_key_path.chmod(0o644)
            
            self.logger.info("Created new RSA key pair for JWT signing")
            return private_key, public_key
            
        except Exception as e:
            self.logger.error(f"Failed to create RSA keys: {e}")
            raise
    
    def _start_cleanup_task(self) -> None:
        """Start background cleanup task"""
        async def cleanup_loop():
            while True:
                try:
                    await asyncio.sleep(self.cleanup_interval.total_seconds())
                    await self.cleanup_expired_sessions()
                except Exception as e:
                    self.logger.error(f"Error in session cleanup task: {e}")
        
        self._cleanup_task = asyncio.create_task(cleanup_loop())
        self.logger.info("Session cleanup task started")
    
    async def shutdown(self) -> None:
        """Shutdown session manager gracefully"""
        self.logger.info("Shutting down session manager...")
        
        # Cancel cleanup task
        if self._cleanup_task and not self._cleanup_task.done():
            self._cleanup_task.cancel()
            try:
                await self._cleanup_task
            except asyncio.CancelledError:
                pass
        
        # Invalidate all active sessions
        async with self._session_lock:
            active_tokens = [
                token for token, session in self.active_sessions.items()
                if session.is_active
            ]
            
            for token in active_tokens:
                await self._invalidate_session_internal(token, "system_shutdown")
        
        self.logger.info("Session manager shutdown complete")