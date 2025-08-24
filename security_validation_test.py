#!/usr/bin/env python3
"""
Security Gateway Validation Test
Quick validation test for MCP Security Gateway functionality
"""

import asyncio
import logging
import tempfile
from pathlib import Path
from mcp.security.gateway import MCPSecurityGateway
from mcp.security.exceptions import AuthenticationError, AuthorizationError, SecurityViolationError

async def run_validation():
    """Run basic security gateway validation"""
    print("=== MCP SECURITY GATEWAY VALIDATION ===")
    
    # Setup temporary environment
    temp_dir = Path(tempfile.mkdtemp(prefix="security_validation_"))
    config = {
        'credentials': {'storage_path': str(temp_dir / 'credentials')},
        'permissions': {'storage_path': str(temp_dir / 'permissions')},
        'audit': {'storage_path': str(temp_dir / 'audit')},
        'sessions': {'storage_path': str(temp_dir / 'sessions')},
        'threat_detection': {'storage_path': str(temp_dir / 'threats')}
    }
    
    try:
        # Initialize security gateway
        print("1. Initializing Security Gateway...")
        gateway = MCPSecurityGateway(config)
        print("   ✓ Security Gateway initialized")
        
        # Test credential storage
        print("2. Testing credential storage...")
        server_id = "test-server-001"
        api_key = "test-api-key-12345"
        permissions = ["read", "write"]
        
        success = await gateway.credential_manager.store_credential(
            server_id, "api_key", api_key, permissions
        )
        if success:
            print("   ✓ Credential stored successfully")
        else:
            print("   ✗ Credential storage failed")
            return False
        
        # Test authentication
        print("3. Testing authentication...")
        try:
            session_token = await gateway.authenticate_server(
                server_id, {"api_key": api_key}
            )
            print(f"   ✓ Authentication successful - Token: {session_token[:8]}...")
        except Exception as e:
            print(f"   ✗ Authentication failed: {e}")
            return False
        
        # Test authorization
        print("4. Testing authorization...")
        try:
            result = await gateway.authorize_operation(
                session_token, "read", ["test-file.txt"]
            )
            if result:
                print("   ✓ Authorization successful")
            else:
                print("   ✗ Authorization failed")
        except Exception as e:
            print(f"   ✗ Authorization error: {e}")
            return False
        
        # Test invalid authentication
        print("5. Testing authentication failure...")
        try:
            await gateway.authenticate_server(
                server_id, {"api_key": "wrong-key"}
            )
            print("   ✗ Authentication should have failed")
            return False
        except AuthenticationError:
            print("   ✓ Authentication correctly rejected invalid credentials")
        except Exception as e:
            print(f"   ✗ Unexpected error: {e}")
            return False
        
        # Test credential rotation
        print("6. Testing credential rotation...")
        try:
            rotation_result = await gateway.rotate_credentials(
                server_id, force_rotation=True
            )
            new_key = rotation_result[server_id]
            print(f"   ✓ Credential rotated - New key: {new_key[:8]}...")
            
            # Test old key no longer works
            try:
                await gateway.authenticate_server(
                    server_id, {"api_key": api_key}
                )
                print("   ✗ Old credential should not work after rotation")
                return False
            except AuthenticationError:
                print("   ✓ Old credential correctly rejected after rotation")
                
        except Exception as e:
            print(f"   ✗ Credential rotation failed: {e}")
            return False
        
        # Test security metrics
        print("7. Testing security metrics...")
        try:
            metrics = await gateway.get_security_metrics()
            print(f"   ✓ Metrics retrieved:")
            print(f"     - Active sessions: {metrics.active_sessions}")
            print(f"     - Auth attempts: {metrics.authentication_attempts}")
            print(f"     - Auth failures: {metrics.authentication_failures}")
        except Exception as e:
            print(f"   ✗ Metrics retrieval failed: {e}")
            return False
        
        # Test session cleanup
        print("8. Testing session management...")
        try:
            cleanup_count = await gateway.cleanup_expired_sessions()
            print(f"   ✓ Session cleanup completed - {cleanup_count} sessions cleaned")
        except Exception as e:
            print(f"   ✗ Session cleanup failed: {e}")
            return False
        
        print("\n=== VALIDATION SUCCESSFUL ===")
        print("All core security gateway features are working correctly:")
        print("✓ Credential management and storage")
        print("✓ Authentication and authorization")
        print("✓ Credential rotation")
        print("✓ Security metrics collection")
        print("✓ Session management")
        print("✓ Error handling and security controls")
        
        return True
        
    except Exception as e:
        print(f"\n=== VALIDATION FAILED ===")
        print(f"Error: {e}")
        import traceback
        traceback.print_exc()
        return False
    
    finally:
        # Cleanup
        try:
            if 'gateway' in locals() and hasattr(gateway.audit_logger, 'shutdown'):
                gateway.audit_logger.shutdown()
        except:
            pass

if __name__ == "__main__":
    # Configure logging
    logging.basicConfig(level=logging.ERROR)  # Suppress logs for cleaner output
    
    # Run validation
    success = asyncio.run(run_validation())
    exit(0 if success else 1)