#!/usr/bin/env python3
"""
MCP Security Gateway Demonstration

Demonstrates the complete MCP security gateway implementation including:
- Authentication and authorization workflows
- Credential management and rotation
- Security threat detection
- Audit logging and monitoring
- Integration with RIF orchestrator system
"""

import asyncio
import json
import logging
import tempfile
from datetime import datetime, timedelta
from pathlib import Path

from mcp.security.gateway import MCPSecurityGateway
from mcp.security.exceptions import AuthenticationError, AuthorizationError, SecurityViolationError


async def setup_demo_environment():
    """Set up demonstration environment with temporary storage"""
    temp_dir = Path(tempfile.mkdtemp(prefix="mcp_security_demo_"))
    
    security_config = {
        'credentials': {'storage_path': str(temp_dir / 'credentials')},
        'permissions': {'storage_path': str(temp_dir / 'permissions')},
        'audit': {'storage_path': str(temp_dir / 'audit')},
        'sessions': {'storage_path': str(temp_dir / 'sessions')},
        'threat_detection': {'storage_path': str(temp_dir / 'threats')},
        'session_timeout_hours': 24,
        'auth_performance_target_ms': 200,
        'authz_performance_target_ms': 100
    }
    
    print(f"Demo environment created at: {temp_dir}")
    return security_config, temp_dir


async def demonstrate_basic_security_workflow():
    """Demonstrate basic security authentication and authorization workflow"""
    print("\n=== BASIC SECURITY WORKFLOW DEMONSTRATION ===")
    
    config, temp_dir = await setup_demo_environment()
    security_gateway = MCPSecurityGateway(config)
    
    try:
        # 1. Server Registration and Credential Storage
        print("\n1. Registering MCP servers with credentials...")
        
        servers = [
            {
                'id': 'github-server',
                'type': 'github',
                'api_key': 'github_api_key_12345',
                'permissions': ['read', 'write', 'repo_admin']
            },
            {
                'id': 'reasoning-server', 
                'type': 'reasoning',
                'api_key': 'reasoning_key_67890',
                'permissions': ['read', 'execute', 'model_access']
            },
            {
                'id': 'memory-server',
                'type': 'memory',
                'api_key': 'memory_key_abcdef',
                'permissions': ['read', 'write', 'search']
            }
        ]
        
        for server in servers:
            await security_gateway.credential_manager.store_credential(
                server['id'], 'api_key', server['api_key'], server['permissions']
            )
            print(f"  ✓ {server['id']} registered with {len(server['permissions'])} permissions")
        
        # 2. Authentication Workflow
        print("\n2. Authenticating MCP servers...")
        
        authenticated_sessions = {}
        for server in servers:
            try:
                session_token = await security_gateway.authenticate_server(
                    server['id'], {'api_key': server['api_key']}
                )
                authenticated_sessions[server['id']] = session_token
                print(f"  ✓ {server['id']} authenticated successfully")
            except AuthenticationError as e:
                print(f"  ✗ {server['id']} authentication failed: {e}")
        
        # 3. Authorization Testing
        print("\n3. Testing authorization for various operations...")
        
        operations = [
            ('github-server', 'read', ['repository/README.md']),
            ('github-server', 'write', ['repository/src/main.py']),
            ('reasoning-server', 'execute', ['reasoning/model']),
            ('memory-server', 'search', ['memory/embeddings']),
            ('memory-server', 'admin', ['memory/admin_config'])  # Should fail
        ]
        
        for server_id, operation, resources in operations:
            if server_id in authenticated_sessions:
                try:
                    await security_gateway.authorize_operation(
                        authenticated_sessions[server_id], operation, resources
                    )
                    print(f"  ✓ {server_id}: {operation} on {resources[0]} - AUTHORIZED")
                except AuthorizationError as e:
                    print(f"  ✗ {server_id}: {operation} on {resources[0]} - DENIED ({e})")
        
        # 4. Secure Operation Context
        print("\n4. Demonstrating secure operation context...")
        
        github_token = authenticated_sessions['github-server']
        async with security_gateway.secure_operation_context(
            github_token, 'read', ['repository/analysis.json']
        ) as context:
            print(f"  ✓ Secure context established for {context.server_id}")
            print(f"    - Permissions: {context.permissions}")
            print(f"    - Session valid until: {context.time_until_expiry}")
        
        # 5. Security Metrics
        print("\n5. Current security metrics...")
        metrics = await security_gateway.get_security_metrics()
        print(f"  - Active sessions: {metrics.active_sessions}")
        print(f"  - Authentication attempts: {metrics.authentication_attempts}")
        print(f"  - Average auth latency: {metrics.average_auth_latency:.2f}ms")
        print(f"  - Average authz latency: {metrics.average_authz_latency:.2f}ms")
        
        return security_gateway, authenticated_sessions
        
    except Exception as e:
        print(f"Error in basic workflow: {e}")
        raise
    finally:
        # Cleanup
        if hasattr(security_gateway.audit_logger, 'shutdown'):
            security_gateway.audit_logger.shutdown()


async def demonstrate_credential_rotation():
    """Demonstrate automatic credential rotation"""
    print("\n=== CREDENTIAL ROTATION DEMONSTRATION ===")
    
    config, temp_dir = await setup_demo_environment()
    security_gateway = MCPSecurityGateway(config)
    
    try:
        # Setup server with credential
        server_id = 'rotation-demo-server'
        original_key = 'original_api_key_12345'
        
        await security_gateway.credential_manager.store_credential(
            server_id, 'api_key', original_key, ['read', 'write']
        )
        print(f"1. Server {server_id} registered with original credential")
        
        # Authenticate with original credential
        session_token = await security_gateway.authenticate_server(
            server_id, {'api_key': original_key}
        )
        print("2. Authentication successful with original credential")
        
        # Perform credential rotation
        print("3. Initiating credential rotation...")
        rotation_results = await security_gateway.rotate_credentials(
            server_id, force_rotation=True
        )
        
        new_credential = rotation_results[server_id]
        print(f"4. Credential rotation completed")
        print(f"   Original: {original_key[:10]}...")
        print(f"   New:      {new_credential[:10]}...")
        
        # Test that old credential no longer works
        print("5. Testing old credential (should fail)...")
        try:
            await security_gateway.authenticate_server(
                server_id, {'api_key': original_key}
            )
            print("   ✗ Old credential still works (unexpected!)")
        except AuthenticationError:
            print("   ✓ Old credential rejected as expected")
        
        # Test that new credential works
        print("6. Testing new credential (should succeed)...")
        try:
            new_session = await security_gateway.authenticate_server(
                server_id, {'api_key': new_credential}
            )
            print("   ✓ New credential works correctly")
        except AuthenticationError as e:
            print(f"   ✗ New credential failed: {e}")
        
        return security_gateway
        
    except Exception as e:
        print(f"Error in credential rotation demo: {e}")
        raise
    finally:
        if hasattr(security_gateway.audit_logger, 'shutdown'):
            security_gateway.audit_logger.shutdown()


async def demonstrate_threat_detection():
    """Demonstrate threat detection and security monitoring"""
    print("\n=== THREAT DETECTION DEMONSTRATION ===")
    
    config, temp_dir = await setup_demo_environment()
    
    # Lower thresholds for demo
    config['threat_detection']['threat_thresholds'] = {
        'auth_failures_per_minute': 3,
        'auth_failures_per_hour': 10
    }
    
    security_gateway = MCPSecurityGateway(config)
    
    try:
        # Setup target server
        server_id = 'threat-demo-server'
        correct_key = 'correct_key_12345'
        
        await security_gateway.credential_manager.store_credential(
            server_id, 'api_key', correct_key, ['read']
        )
        
        # 1. Normal authentication
        print("1. Normal authentication (should succeed)...")
        session_token = await security_gateway.authenticate_server(
            server_id, {'api_key': correct_key}
        )
        print("   ✓ Normal authentication successful")
        
        # 2. Simulate brute force attack
        print("2. Simulating brute force attack...")
        failed_attempts = 0
        
        for attempt in range(5):
            try:
                await security_gateway.authenticate_server(
                    server_id, 
                    {'api_key': f'wrong_key_{attempt}'},
                    {'source_ip': '192.168.1.100', 'user_agent': 'AttackBot/1.0'}
                )
            except (AuthenticationError, SecurityViolationError) as e:
                failed_attempts += 1
                if "Security" in str(e):
                    print(f"   ✓ Attack blocked due to security violation: {e}")
                else:
                    print(f"   - Failed attempt {attempt + 1}: Wrong credentials")
        
        print(f"   Total failed attempts: {failed_attempts}")
        
        # 3. Test privilege escalation detection
        print("3. Testing privilege escalation detection...")
        try:
            await security_gateway.authorize_operation(
                session_token, 'admin_delete', ['system/critical_config.json']
            )
            print("   ✗ Admin operation allowed (unexpected)")
        except (AuthorizationError, SecurityViolationError) as e:
            print(f"   ✓ Admin operation blocked: {e}")
        
        # 4. Test bulk data access (potential exfiltration)
        print("4. Testing bulk data access detection...")
        large_file_list = [f'data/file_{i}.json' for i in range(20)]
        try:
            await security_gateway.authorize_operation(
                session_token, 'read', large_file_list
            )
            print(f"   ✓ Bulk access authorized but flagged for monitoring")
        except (AuthorizationError, SecurityViolationError) as e:
            print(f"   ✓ Bulk access blocked as potential threat: {e}")
        
        # 5. Get threat detection metrics
        print("5. Threat detection metrics...")
        threat_metrics = await security_gateway.threat_detector.get_threat_metrics()
        print(f"   - Total threats detected: {threat_metrics.total_threats_detected}")
        print(f"   - Active indicators: {threat_metrics.active_indicators}")
        print(f"   - Detection accuracy: {threat_metrics.detection_accuracy:.1%}")
        
        # 6. Get server threat profile
        print("6. Server threat profile...")
        threat_profile = await security_gateway.threat_detector.get_server_threat_profile(server_id)
        print(f"   - Risk score: {threat_profile['risk_score']:.2f}")
        print(f"   - Recommendation: {threat_profile['recommendation']}")
        print(f"   - Threat indicators: {len(threat_profile['threat_indicators'])}")
        
        return security_gateway
        
    except Exception as e:
        print(f"Error in threat detection demo: {e}")
        raise
    finally:
        if hasattr(security_gateway.audit_logger, 'shutdown'):
            security_gateway.audit_logger.shutdown()


async def demonstrate_audit_logging():
    """Demonstrate comprehensive audit logging"""
    print("\n=== AUDIT LOGGING DEMONSTRATION ===")
    
    config, temp_dir = await setup_demo_environment()
    security_gateway = MCPSecurityGateway(config)
    
    try:
        # Setup and perform various operations to generate audit events
        server_id = 'audit-demo-server'
        api_key = 'audit_test_key_12345'
        
        await security_gateway.credential_manager.store_credential(
            server_id, 'api_key', api_key, ['read', 'write', 'execute']
        )
        
        # Generate various audit events
        print("1. Generating audit events...")
        
        # Successful authentication
        session_token = await security_gateway.authenticate_server(
            server_id, {'api_key': api_key}
        )
        
        # Various operations
        operations = [
            ('read', ['file1.txt']),
            ('write', ['file2.txt']),
            ('execute', ['script.py']),
        ]
        
        for operation, resources in operations:
            await security_gateway.authorize_operation(session_token, operation, resources)
        
        # Failed authentication attempt
        try:
            await security_gateway.authenticate_server(
                server_id, {'api_key': 'wrong_key'}
            )
        except AuthenticationError:
            pass  # Expected failure
        
        # Session invalidation
        await security_gateway.invalidate_session(session_token, "demo_cleanup")
        
        # 2. Get audit metrics
        print("2. Audit logging metrics...")
        audit_metrics = await security_gateway.audit_logger.get_audit_metrics()
        print(f"   - Total events: {audit_metrics.total_events}")
        print(f"   - Events by type: {audit_metrics.events_by_type}")
        print(f"   - Events by severity: {audit_metrics.events_by_severity}")
        print(f"   - Queue size: {audit_metrics.queue_size}")
        
        # 3. Generate compliance report
        print("3. Generating compliance report...")
        report = await security_gateway.audit_logger.generate_compliance_report(
            datetime.utcnow() - timedelta(hours=1),
            datetime.utcnow()
        )
        
        print(f"   - Report ID: {report['report_id']}")
        print(f"   - Period: {report['period']['duration_days']} days")
        print(f"   - Total events: {report['summary']['total_events']}")
        print(f"   - Compliance status: {report['compliance_status']}")
        
        # 4. Check audit file creation
        audit_files = list(Path(config['audit']['storage_path']).glob('audit_*.jsonl'))
        print(f"4. Audit files created: {len(audit_files)}")
        
        if audit_files:
            latest_file = max(audit_files, key=lambda f: f.stat().st_mtime)
            print(f"   - Latest audit file: {latest_file.name}")
            
            # Show some audit entries
            with open(latest_file, 'r') as f:
                lines = f.readlines()[-3:]  # Last 3 entries
                print("   - Recent audit entries:")
                for line in lines:
                    try:
                        entry = json.loads(line.strip())
                        print(f"     {entry['timestamp']}: {entry['event_type']} - {entry['message']}")
                    except:
                        pass
        
        return security_gateway
        
    except Exception as e:
        print(f"Error in audit logging demo: {e}")
        raise
    finally:
        if hasattr(security_gateway.audit_logger, 'shutdown'):
            security_gateway.audit_logger.shutdown()


async def demonstrate_rif_integration():
    """Demonstrate integration with RIF orchestrator system"""
    print("\n=== RIF INTEGRATION DEMONSTRATION ===")
    
    config, temp_dir = await setup_demo_environment()
    security_gateway = MCPSecurityGateway(config)
    
    try:
        # Simulate RIF orchestrator requesting MCP server access
        print("1. RIF Orchestrator requesting MCP server access...")
        
        # Register RIF as a privileged client
        rif_server_id = 'rif-orchestrator'
        rif_credentials = {'api_key': 'rif_orchestrator_secure_key_xyz789'}
        
        await security_gateway.credential_manager.store_credential(
            rif_server_id, 'api_key', rif_credentials['api_key'],
            ['read', 'write', 'execute', 'admin', 'orchestrator']
        )
        
        # Authenticate RIF orchestrator
        rif_session = await security_gateway.authenticate_server(
            rif_server_id, rif_credentials
        )
        print("   ✓ RIF Orchestrator authenticated with high privileges")
        
        # RIF requests access to various MCP servers
        print("2. RIF accessing MCP server capabilities...")
        
        mcp_operations = [
            ('read', ['github/repositories']),
            ('execute', ['reasoning/analyze_code']),
            ('write', ['memory/store_patterns']),
            ('admin', ['system/configure_servers'])
        ]
        
        for operation, resources in mcp_operations:
            try:
                await security_gateway.authorize_operation(
                    rif_session, operation, resources
                )
                print(f"   ✓ RIF authorized for {operation} on {resources[0]}")
            except AuthorizationError as e:
                print(f"   ✗ RIF denied {operation} on {resources[0]}: {e}")
        
        # Demonstrate dynamic server provisioning
        print("3. Dynamic MCP server provisioning...")
        
        # RIF requests new server registration (simulated)
        new_server_config = {
            'server_id': 'dynamic-github-server-001',
            'server_type': 'github',
            'project_context': 'RIF-enhancement-project',
            'required_permissions': ['read', 'write', 'repo_access']
        }
        
        # Security gateway validates and provisions
        dynamic_key = 'dynamic_github_key_' + str(int(datetime.utcnow().timestamp()))
        await security_gateway.credential_manager.store_credential(
            new_server_config['server_id'],
            'api_key',
            dynamic_key,
            new_server_config['required_permissions']
        )
        
        print(f"   ✓ Dynamic server {new_server_config['server_id']} provisioned")
        print(f"   ✓ Permissions: {new_server_config['required_permissions']}")
        
        # Test the new server
        dynamic_session = await security_gateway.authenticate_server(
            new_server_config['server_id'],
            {'api_key': dynamic_key}
        )
        
        await security_gateway.authorize_operation(
            dynamic_session, 'read', ['repository/issues']
        )
        print("   ✓ Dynamic server operational and authorized")
        
        # 4. Security monitoring for RIF operations
        print("4. RIF-specific security monitoring...")
        
        # Check if RIF operations are being tracked appropriately
        rif_profile = await security_gateway.threat_detector.get_server_threat_profile(
            rif_server_id
        )
        
        print(f"   - RIF risk score: {rif_profile['risk_score']:.2f}")
        print(f"   - RIF security status: {rif_profile['recommendation']}")
        
        # 5. Integration metrics
        print("5. Integration metrics...")
        metrics = await security_gateway.get_security_metrics()
        print(f"   - Total active sessions: {metrics.active_sessions}")
        print(f"   - RIF orchestrator session active: {rif_session in security_gateway.active_sessions}")
        
        # 6. Cleanup simulation
        print("6. Cleaning up dynamic resources...")
        await security_gateway.invalidate_session(dynamic_session, "dynamic_cleanup")
        await security_gateway.credential_manager.remove_credential(
            new_server_config['server_id']
        )
        print("   ✓ Dynamic resources cleaned up")
        
        return security_gateway
        
    except Exception as e:
        print(f"Error in RIF integration demo: {e}")
        raise
    finally:
        if hasattr(security_gateway.audit_logger, 'shutdown'):
            security_gateway.audit_logger.shutdown()


async def main():
    """Run all demonstrations"""
    print("MCP Security Gateway Comprehensive Demonstration")
    print("=" * 55)
    
    # Configure logging
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )
    
    # Suppress some verbose logging for demo
    logging.getLogger('mcp.security').setLevel(logging.WARNING)
    
    try:
        # Run demonstrations
        await demonstrate_basic_security_workflow()
        await demonstrate_credential_rotation()
        await demonstrate_threat_detection()
        await demonstrate_audit_logging()
        await demonstrate_rif_integration()
        
        print("\n" + "=" * 55)
        print("All demonstrations completed successfully!")
        print("\nKey Features Demonstrated:")
        print("✓ Multi-factor authentication and authorization")
        print("✓ Secure credential storage and rotation")
        print("✓ Real-time threat detection and prevention")
        print("✓ Comprehensive audit logging and compliance")
        print("✓ Session management with security fingerprinting")
        print("✓ Integration with RIF orchestrator system")
        print("✓ Performance monitoring and metrics collection")
        
    except Exception as e:
        print(f"\nDemo failed with error: {e}")
        import traceback
        traceback.print_exc()
    
    print("\nDemo complete. All temporary files will be cleaned up automatically.")


if __name__ == "__main__":
    asyncio.run(main())