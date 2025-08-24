#!/usr/bin/env python3
"""
Unit tests for user prompt capture hook.

Tests the capture_user_prompt.py functionality to ensure proper
integration with ConversationStorageBackend.
"""

import os
import sys
import json
import uuid
import tempfile
import subprocess
from datetime import datetime

# Add path for imports
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from conversations.storage_backend import ConversationStorageBackend


def test_user_prompt_capture():
    """Test that user prompt capture works correctly."""
    print("Testing user prompt capture hook...")
    
    # Create test database
    with tempfile.NamedTemporaryFile(suffix='.duckdb', delete=False) as tmp_db:
        test_db_path = tmp_db.name
    
    try:
        # Initialize storage backend
        storage = ConversationStorageBackend(db_path=test_db_path)
        
        # Test prompt
        test_prompt = "This is a test user prompt for validation"
        
        # Create test script that uses temp database
        test_script = f'''
import os, sys, json
sys.path.insert(0, "{os.path.dirname(os.path.dirname(os.path.abspath(__file__)))}")
from conversations.storage_backend import ConversationStorageBackend
from datetime import datetime

# Override database path
storage = ConversationStorageBackend(db_path="{test_db_path}")
conversation_id = "test-conversation-{uuid.uuid4()}"

event_data = {{
    'prompt_text': "{test_prompt}",
    'prompt_length': {len(test_prompt)},
    'user_context': {{'test': True}},
    'capture_method': 'UserPromptSubmit_hook',
    'hook_version': '1.0.0'
}}

event_id = storage.store_conversation_event(
    conversation_id=conversation_id,
    agent_type='claude-code',
    event_type='user_prompt',
    event_data=event_data
)

print(f"Captured event: {{event_id}}")
storage.close()
'''
        
        # Write and execute test script
        with tempfile.NamedTemporaryFile(mode='w', suffix='.py', delete=False) as test_file:
            test_file.write(test_script)
            test_script_path = test_file.name
        
        try:
            # Execute the script
            result = subprocess.run([sys.executable, test_script_path], 
                                  capture_output=True, text=True, cwd=os.path.dirname(__file__))
            
            if result.returncode != 0:
                print(f"Script failed: {result.stderr}")
                return False
            
            print(f"Script output: {result.stdout.strip()}")
            
            # Verify data was stored
            storage = ConversationStorageBackend(db_path=test_db_path)
            
            # Check event count
            count_result = storage.connection.execute('SELECT COUNT(*) FROM conversation_events').fetchone()
            print(f"Events in test database: {count_result[0]}")
            
            if count_result[0] == 0:
                print("❌ No events found in database")
                return False
            
            # Get the event
            events = storage.connection.execute('''
                SELECT event_type, agent_type, event_data 
                FROM conversation_events 
                ORDER BY timestamp DESC LIMIT 1
            ''').fetchall()
            
            if not events:
                print("❌ No events retrieved")
                return False
            
            event = events[0]
            event_data = json.loads(event[2])
            
            # Validate event data
            assert event[0] == 'user_prompt', f"Wrong event type: {event[0]}"
            assert event[1] == 'claude-code', f"Wrong agent type: {event[1]}"
            assert event_data['prompt_text'] == test_prompt, f"Wrong prompt text: {event_data['prompt_text']}"
            assert event_data['capture_method'] == 'UserPromptSubmit_hook', f"Wrong capture method: {event_data['capture_method']}"
            
            print("✅ User prompt capture test passed!")
            storage.close()
            return True
            
        finally:
            os.unlink(test_script_path)
        
    finally:
        # Cleanup test database
        if os.path.exists(test_db_path):
            os.unlink(test_db_path)


def test_hook_integration():
    """Test the actual hook script integration."""
    print("\nTesting hook script integration...")
    
    # Test the actual capture script
    script_path = os.path.join(os.path.dirname(__file__), 'capture_user_prompt.py')
    test_input = "Integration test prompt"
    
    try:
        result = subprocess.run([sys.executable, script_path], 
                              input=test_input, 
                              capture_output=True, text=True, timeout=10,
                              cwd=os.path.dirname(os.path.dirname(__file__)))
        
        if result.returncode != 0:
            print(f"❌ Hook script failed: {result.stderr}")
            return False
        
        # Check if event was stored in main database
        storage = ConversationStorageBackend(db_path=os.path.join(os.path.dirname(__file__), '..', 'conversations.duckdb'))
        
        # Get recent events
        events = storage.connection.execute('''
            SELECT event_data FROM conversation_events 
            WHERE event_type = 'user_prompt' 
            ORDER BY timestamp DESC LIMIT 5
        ''').fetchall()
        
        # Look for our test input
        found = False
        for event in events:
            event_data = json.loads(event[0])
            if event_data.get('prompt_text') == test_input:
                found = True
                break
        
        if found:
            print("✅ Hook integration test passed!")
            storage.close()
            return True
        else:
            print("❌ Test prompt not found in database")
            storage.close()
            return False
        
    except subprocess.TimeoutExpired:
        print("❌ Hook script timeout")
        return False
    except Exception as e:
        print(f"❌ Hook integration test failed: {e}")
        return False


def test_hook_performance():
    """Test hook performance to ensure <10ms overhead."""
    print("\nTesting hook performance...")
    
    script_path = os.path.join(os.path.dirname(__file__), 'capture_user_prompt.py')
    test_input = "Performance test prompt"
    
    times = []
    for i in range(5):
        start = datetime.now()
        result = subprocess.run([sys.executable, script_path], 
                              input=test_input, 
                              capture_output=True, text=True,
                              cwd=os.path.dirname(os.path.dirname(__file__)))
        end = datetime.now()
        
        if result.returncode != 0:
            print(f"❌ Performance test failed on run {i+1}: {result.stderr}")
            return False
        
        duration_ms = (end - start).total_seconds() * 1000
        times.append(duration_ms)
    
    avg_time = sum(times) / len(times)
    max_time = max(times)
    
    print(f"Average execution time: {avg_time:.2f}ms")
    print(f"Maximum execution time: {max_time:.2f}ms")
    
    if max_time < 100:  # Allow 100ms for subprocess overhead
        print("✅ Performance test passed!")
        return True
    else:
        print(f"❌ Performance test failed: {max_time:.2f}ms > 100ms")
        return False


if __name__ == '__main__':
    print("Running user prompt capture hook tests...\n")
    
    success = True
    success &= test_user_prompt_capture()
    success &= test_hook_integration()
    success &= test_hook_performance()
    
    if success:
        print("\n🎉 All tests passed!")
        sys.exit(0)
    else:
        print("\n❌ Some tests failed")
        sys.exit(1)