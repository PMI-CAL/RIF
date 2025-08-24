#!/usr/bin/env python3
"""
Test suite for conversation capture hooks.

Validates that the capture hooks for user prompts, tool usage, and assistant 
responses are working correctly and storing data in the conversation system.
"""

import os
import sys
import tempfile
import subprocess
import json
import time
from datetime import datetime
import unittest

# Add knowledge directory to path
sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 'knowledge'))

from conversations.storage_backend import ConversationStorageBackend


class TestConversationCaptureHooks(unittest.TestCase):
    """Test cases for conversation capture hooks."""
    
    def setUp(self):
        """Set up test environment."""
        # Create temporary database for testing
        fd, self.temp_db_path = tempfile.mkstemp(suffix='.duckdb')
        os.close(fd)
        os.unlink(self.temp_db_path)  # Remove empty file
        
        self.storage = ConversationStorageBackend(self.temp_db_path)
        self.project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
        
        # Store original database path and replace with test database
        self.original_db_path = os.path.join(self.project_root, 'knowledge', 'conversations.duckdb')
        self.test_db_path = self.temp_db_path
        
        # Clean up any existing session files
        session_file = '/tmp/claude_code_conversation_id'
        if os.path.exists(session_file):
            os.unlink(session_file)
    
    def tearDown(self):
        """Clean up test environment."""
        self.storage.close()
        if os.path.exists(self.temp_db_path):
            os.unlink(self.temp_db_path)
        
        # Clean up session files
        session_file = '/tmp/claude_code_conversation_id'
        if os.path.exists(session_file):
            os.unlink(session_file)
    
    def _run_capture_script(self, script_name, input_text, env_vars=None):
        """
        Run a capture script with input and environment variables.
        
        Args:
            script_name: Name of the capture script
            input_text: Text to pass to stdin
            env_vars: Additional environment variables
            
        Returns:
            Subprocess result
        """
        script_path = os.path.join(self.project_root, 'knowledge', 'conversations', script_name)
        
        # Set up environment
        env = os.environ.copy()
        if env_vars:
            env.update(env_vars)
        
        # Replace database path in script temporarily
        # This is a bit hacky but works for testing
        with open(script_path, 'r') as f:
            script_content = f.read()
        
        # Create temporary script with test database path
        temp_script_content = script_content.replace(
            'os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "conversations.duckdb")',
            f'"{self.test_db_path}"'
        )
        
        fd, temp_script_path = tempfile.mkstemp(suffix='.py')
        try:
            with os.fdopen(fd, 'w') as f:
                f.write(temp_script_content)
            
            result = subprocess.run(
                [sys.executable, temp_script_path],
                input=input_text,
                text=True,
                capture_output=True,
                env=env,
                cwd=self.project_root
            )
            
            return result
            
        finally:
            if os.path.exists(temp_script_path):
                os.unlink(temp_script_path)
    
    def test_user_prompt_capture(self):
        """Test user prompt capture hook."""
        # Run user prompt capture
        result = self._run_capture_script(
            'capture_user_prompt.py',
            'Test user prompt for conversation capture'
        )
        
        self.assertEqual(result.returncode, 0, f"Script failed: {result.stderr}")
        
        # Verify data was stored
        events = self.storage.connection.execute("""
            SELECT * FROM conversation_events 
            WHERE event_type = 'user_prompt'
            ORDER BY timestamp DESC
            LIMIT 1
        """).fetchall()
        
        self.assertEqual(len(events), 1, "User prompt event should be stored")
        
        event = events[0]
        event_data = json.loads(event[4])  # event_data column
        
        self.assertEqual(event_data['prompt_text'], 'Test user prompt for conversation capture')
        self.assertEqual(event_data['capture_method'], 'UserPromptSubmit_hook')
        self.assertIn('hook_version', event_data)
    
    def test_tool_use_capture(self):
        """Test tool use capture hook."""
        # Set up environment variables to simulate tool execution
        env_vars = {
            'CLAUDE_TOOL_NAME': 'Read',
            'CLAUDE_TOOL_PARAMS': json.dumps({
                'file_path': '/test/file.py',
                'limit': 100
            }),
            'CLAUDE_TOOL_EXIT_CODE': '0',
            'CLAUDE_TOOL_STDOUT': 'File contents here...',
            'CLAUDE_TOOL_START_TIME': str(time.time() - 1),
            'CLAUDE_TOOL_END_TIME': str(time.time())
        }
        
        result = self._run_capture_script(
            'capture_tool_use.py',
            '',  # Tool capture doesn't read stdin
            env_vars
        )
        
        self.assertEqual(result.returncode, 0, f"Script failed: {result.stderr}")
        
        # Verify data was stored
        events = self.storage.connection.execute("""
            SELECT * FROM conversation_events 
            WHERE event_type = 'tool_success'
            ORDER BY timestamp DESC
            LIMIT 1
        """).fetchall()
        
        self.assertEqual(len(events), 1, "Tool use event should be stored")
        
        event = events[0]
        event_data = json.loads(event[4])  # event_data column
        
        self.assertEqual(event_data['tool_name'], 'Read')
        self.assertEqual(event_data['execution_status'], 'success')
        self.assertEqual(event_data['capture_method'], 'ToolUse_hook')
        self.assertIn('tool_params', event_data)
        self.assertEqual(event_data['tool_params']['file_path'], '/test/file.py')
    
    def test_tool_use_capture_error(self):
        """Test tool use capture for failed tool execution."""
        env_vars = {
            'CLAUDE_TOOL_NAME': 'Write',
            'CLAUDE_TOOL_PARAMS': json.dumps({
                'file_path': '/readonly/file.py',
                'content': 'test content'
            }),
            'CLAUDE_TOOL_EXIT_CODE': '1',
            'CLAUDE_TOOL_STDERR': 'Permission denied: cannot write to /readonly/file.py',
            'CLAUDE_TOOL_ERROR': 'PermissionError: Permission denied'
        }
        
        result = self._run_capture_script(
            'capture_tool_use.py',
            '',
            env_vars
        )
        
        self.assertEqual(result.returncode, 0, f"Script failed: {result.stderr}")
        
        # Verify error event was stored
        events = self.storage.connection.execute("""
            SELECT * FROM conversation_events 
            WHERE event_type = 'tool_error'
            ORDER BY timestamp DESC
            LIMIT 1
        """).fetchall()
        
        self.assertEqual(len(events), 1, "Tool error event should be stored")
        
        # Verify error was also stored in conversation_errors table
        errors = self.storage.connection.execute("""
            SELECT * FROM conversation_errors
            WHERE error_type = 'tool_execution_error'
            ORDER BY timestamp DESC
            LIMIT 1
        """).fetchall()
        
        self.assertEqual(len(errors), 1, "Tool execution error should be stored")
    
    def test_assistant_response_capture(self):
        """Test assistant response capture hook."""
        # Create a realistic assistant response with tool calls
        response_text = """I'll help you implement the feature. Let me start by reading the configuration file.

<function_calls>
<invoke name="Read">
<parameter name="file_path">/config/settings.yaml</parameter>
</invoke>
</function_calls>

Based on the configuration, I need to implement the following:

```python
def implement_feature():
    # Implementation here
    return True
```

Let me also check the existing code structure:

<function_calls>
<invoke name="Glob">
<parameter name="pattern">src/**/*.py</parameter>
</invoke>
</function_calls>

This approach will ensure compatibility with the existing codebase."""
        
        result = self._run_capture_script(
            'capture_assistant_response.py',
            response_text
        )
        
        self.assertEqual(result.returncode, 0, f"Script failed: {result.stderr}")
        
        # Verify data was stored
        events = self.storage.connection.execute("""
            SELECT * FROM conversation_events 
            WHERE event_type = 'assistant_response'
            ORDER BY timestamp DESC
            LIMIT 1
        """).fetchall()
        
        self.assertEqual(len(events), 1, "Assistant response event should be stored")
        
        event = events[0]
        event_data = json.loads(event[4])  # event_data column
        
        self.assertEqual(event_data['capture_method'], 'AssistantResponse_hook')
        self.assertIn('response_analysis', event_data)
        
        analysis = event_data['response_analysis']
        self.assertEqual(analysis['tool_calls_count'], 2)
        self.assertIn('Read', analysis['tools_used'])
        self.assertIn('Glob', analysis['tools_used'])
        self.assertEqual(analysis['code_blocks_count'], 1)
        self.assertIn('action', analysis['response_types'])
        self.assertIn('code', analysis['response_types'])
        
        # Verify decision was also captured for tool usage
        decisions = self.storage.connection.execute("""
            SELECT * FROM agent_decisions
            WHERE decision_point LIKE 'Tool selection%'
            ORDER BY timestamp DESC
            LIMIT 1
        """).fetchall()
        
        self.assertEqual(len(decisions), 1, "Tool selection decision should be captured")
    
    def test_conversation_continuity(self):
        """Test that all captures use the same conversation ID."""
        # Run multiple captures in sequence
        self._run_capture_script(
            'capture_user_prompt.py',
            'Initial user prompt'
        )
        
        time.sleep(0.1)  # Small delay to ensure different timestamps
        
        self._run_capture_script(
            'capture_tool_use.py',
            '',
            {'CLAUDE_TOOL_NAME': 'Read', 'CLAUDE_TOOL_EXIT_CODE': '0'}
        )
        
        time.sleep(0.1)
        
        self._run_capture_script(
            'capture_assistant_response.py',
            'Here is the response after reading the file.'
        )
        
        # Verify all events share the same conversation ID
        conversation_ids = self.storage.connection.execute("""
            SELECT DISTINCT conversation_id FROM conversation_events
        """).fetchall()
        
        self.assertEqual(len(conversation_ids), 1, "All events should share the same conversation ID")
        
        # Verify conversation metadata was created
        metadata = self.storage.connection.execute("""
            SELECT * FROM conversation_metadata
            WHERE conversation_id = ?
        """, [conversation_ids[0][0]]).fetchall()
        
        self.assertEqual(len(metadata), 1, "Conversation metadata should be created")
        
        # Verify event count
        total_events = self.storage.connection.execute("""
            SELECT COUNT(*) FROM conversation_events
        """).fetchone()[0]
        
        self.assertEqual(total_events, 3, "Should have 3 events captured")
    
    def test_hook_error_handling(self):
        """Test that hooks handle errors gracefully."""
        # Test with invalid JSON in tool params
        env_vars = {
            'CLAUDE_TOOL_NAME': 'Read',
            'CLAUDE_TOOL_PARAMS': 'invalid json{',
            'CLAUDE_TOOL_EXIT_CODE': '0'
        }
        
        result = self._run_capture_script(
            'capture_tool_use.py',
            '',
            env_vars
        )
        
        # Should not fail even with invalid JSON
        self.assertEqual(result.returncode, 0, "Script should handle invalid JSON gracefully")
        
        # Verify event was still stored
        events = self.storage.connection.execute("""
            SELECT * FROM conversation_events 
            WHERE event_type = 'tool_success'
            ORDER BY timestamp DESC
            LIMIT 1
        """).fetchall()
        
        self.assertEqual(len(events), 1, "Event should be stored even with invalid params")
        
        event_data = json.loads(events[0][4])
        self.assertIn('raw_params', event_data['tool_params'])


def run_capture_hook_tests():
    """Run all conversation capture hook tests."""
    suite = unittest.TestLoader().loadTestsFromTestCase(TestConversationCaptureHooks)
    runner = unittest.TextTestRunner(verbosity=2)
    result = runner.run(suite)
    return result.wasSuccessful()


if __name__ == '__main__':
    success = run_capture_hook_tests()
    sys.exit(0 if success else 1)