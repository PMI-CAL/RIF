"""
MCP Security Gateway Exceptions

Custom exceptions for the MCP Security Gateway system providing
clear error handling and security event categorization.
"""


class SecurityGatewayError(Exception):
    """Base exception for MCP Security Gateway errors"""
    
    def __init__(self, message: str, error_code: str = None, details: dict = None):
        super().__init__(message)
        self.message = message
        self.error_code = error_code
        self.details = details or {}


class AuthenticationError(SecurityGatewayError):
    """Exception raised when authentication fails"""
    
    def __init__(self, message: str = "Authentication failed", **kwargs):
        super().__init__(message, error_code="AUTH_FAILED", **kwargs)


class AuthorizationError(SecurityGatewayError):
    """Exception raised when authorization fails"""
    
    def __init__(self, message: str = "Authorization failed", **kwargs):
        super().__init__(message, error_code="AUTHZ_FAILED", **kwargs)


class SecurityViolationError(SecurityGatewayError):
    """Exception raised when a security policy violation is detected"""
    
    def __init__(self, message: str = "Security policy violation", **kwargs):
        super().__init__(message, error_code="SECURITY_VIOLATION", **kwargs)


class CredentialError(SecurityGatewayError):
    """Exception raised for credential management errors"""
    
    def __init__(self, message: str = "Credential error", **kwargs):
        super().__init__(message, error_code="CREDENTIAL_ERROR", **kwargs)


class SessionError(SecurityGatewayError):
    """Exception raised for session management errors"""
    
    def __init__(self, message: str = "Session error", **kwargs):
        super().__init__(message, error_code="SESSION_ERROR", **kwargs)


class PermissionError(SecurityGatewayError):
    """Exception raised for permission matrix errors"""
    
    def __init__(self, message: str = "Permission error", **kwargs):
        super().__init__(message, error_code="PERMISSION_ERROR", **kwargs)


class ThreatDetectionError(SecurityGatewayError):
    """Exception raised for threat detection errors"""
    
    def __init__(self, message: str = "Threat detection error", **kwargs):
        super().__init__(message, error_code="THREAT_DETECTION_ERROR", **kwargs)


class AuditError(SecurityGatewayError):
    """Exception raised for audit logging errors"""
    
    def __init__(self, message: str = "Audit logging error", **kwargs):
        super().__init__(message, error_code="AUDIT_ERROR", **kwargs)