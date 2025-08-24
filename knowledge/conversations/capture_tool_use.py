#!/usr/bin/env python3
"""
ToolUse Capture Hook for Claude Code.

Captures tool usage events from Claude Code and stores them in the
conversation system via ConversationStorageBackend.

This hook is triggered during tool execution and captures:
- Tool name and parameters
- Tool execution status and results
- Performance metrics and timing
- Error information if the tool fails

Issue #46: Implement ToolUse capture hook (complexity:low)
"""

import os
import sys
import json
import uuid
import time
from datetime import datetime
from typing import Dict, Any, Optional
import logging

# Add the knowledge directory to path for imports
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

try:
    from conversations.storage_backend import ConversationStorageBackend
except ImportError:
    # Fallback import for direct execution
    from storage_backend import ConversationStorageBackend

# Configure logging to be silent for Claude Code integration
logging.basicConfig(
    level=logging.ERROR,  # Only log errors
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('/Users/cal/DEV/RIF/knowledge/conversations/capture_hooks.log', mode='a'),
    ]
)

logger = logging.getLogger(__name__)


def get_conversation_id() -> str:
    """
    Get or create conversation ID for the current session.
    
    Uses the same session file as the user prompt capture to maintain
    conversation continuity across different hook types.
    
    Returns:
        Conversation ID for current session
    """
    try:
        # Use same session file as user prompt capture
        session_file = '/tmp/claude_code_conversation_id'
        
        # Check if we have an existing conversation ID from this session
        if os.path.exists(session_file):
            with open(session_file, 'r') as f:
                existing_id = f.read().strip()
                # Validate the ID format
                if existing_id and len(existing_id) == 36:  # UUID length
                    return existing_id
        
        # Generate new conversation ID for new session
        conversation_id = str(uuid.uuid4())
        
        # Store for session continuity
        try:
            with open(session_file, 'w') as f:
                f.write(conversation_id)
        except Exception as e:
            logger.warning(f"Failed to store conversation ID: {e}")
        
        return conversation_id
        
    except Exception as e:
        logger.error(f"Failed to generate conversation ID: {e}")
        return str(uuid.uuid4())  # Fallback to new ID


def extract_tool_info() -> Dict[str, Any]:
    """
    Extract tool information from Claude Code environment variables with enhanced parsing.
    
    Claude Code provides tool information through environment variables
    during hook execution.
    
    Returns:
        Dictionary containing tool execution information
    """
    extraction_start = time.time()
    
    tool_info = {
        'timestamp': datetime.now().isoformat(),
        'session_id': os.environ.get('CLAUDE_SESSION_ID', 'unknown'),
        'extraction_time': extraction_start
    }
    
    # Extract Claude Code tool context
    tool_name = os.environ.get('CLAUDE_TOOL_NAME', 'unknown')
    tool_info['tool_name'] = tool_name
    
    # Enhanced tool parameters parsing
    tool_params_str = os.environ.get('CLAUDE_TOOL_PARAMS', '{}')
    try:
        parsed_params = json.loads(tool_params_str)
        tool_info['tool_params'] = parsed_params
        
        # Extract common parameters for indexing
        tool_info['param_keys'] = list(parsed_params.keys()) if isinstance(parsed_params, dict) else []
        tool_info['param_count'] = len(tool_info['param_keys'])
        
    except json.JSONDecodeError:
        tool_info['tool_params'] = {'raw_params': tool_params_str}
        tool_info['param_keys'] = ['raw_params']
        tool_info['param_count'] = 1
        tool_info['parse_error'] = True
    
    # Get tool execution status with enhanced classification
    exit_code = int(os.environ.get('CLAUDE_TOOL_EXIT_CODE', '0'))
    tool_info['exit_code'] = exit_code
    tool_info['success'] = exit_code == 0
    
    # Classify execution result
    if exit_code == 0:
        tool_info['execution_result'] = 'success'
    elif exit_code == 1:
        tool_info['execution_result'] = 'general_error'
    elif exit_code == 2:
        tool_info['execution_result'] = 'misuse_error'
    elif exit_code == 127:
        tool_info['execution_result'] = 'command_not_found'
    elif exit_code == 130:
        tool_info['execution_result'] = 'interrupted'
    else:
        tool_info['execution_result'] = f'error_{exit_code}'
    
    # Get timing information with validation
    start_time = os.environ.get('CLAUDE_TOOL_START_TIME')
    end_time = os.environ.get('CLAUDE_TOOL_END_TIME')
    if start_time and end_time:
        try:
            start_float = float(start_time)
            end_float = float(end_time)
            duration_ms = (end_float - start_float) * 1000
            tool_info['duration_ms'] = round(duration_ms, 2)
            tool_info['start_timestamp'] = start_float
            tool_info['end_timestamp'] = end_float
        except (ValueError, TypeError) as e:
            tool_info['timing_parse_error'] = str(e)
    
    # Enhanced tool output/error processing
    tool_stdout = os.environ.get('CLAUDE_TOOL_STDOUT', '')
    tool_stderr = os.environ.get('CLAUDE_TOOL_STDERR', '')
    tool_error = os.environ.get('CLAUDE_TOOL_ERROR', '')
    
    # Intelligently truncate output based on content
    if tool_stdout:
        stdout_lines = tool_stdout.split('\n')
        if len(tool_stdout) > 2000:
            # Keep first and last lines with indication of truncation
            truncated_stdout = '\n'.join(stdout_lines[:10] + ['... [truncated] ...'] + stdout_lines[-5:])
            tool_info['stdout'] = truncated_stdout
            tool_info['stdout_truncated'] = True
            tool_info['stdout_original_length'] = len(tool_stdout)
        else:
            tool_info['stdout'] = tool_stdout
        tool_info['stdout_line_count'] = len(stdout_lines)
    
    if tool_stderr:
        tool_info['stderr'] = tool_stderr[:2000]  # Keep more stderr as it's usually shorter
        if len(tool_stderr) > 2000:
            tool_info['stderr_truncated'] = True
            tool_info['stderr_original_length'] = len(tool_stderr)
    
    if tool_error:
        tool_info['error'] = tool_error[:1500]  # Error messages are usually concise
        if len(tool_error) > 1500:
            tool_info['error_truncated'] = True
    
    # Enhanced file operation context
    file_operation_tools = ['Read', 'Write', 'Edit', 'MultiEdit', 'LS', 'Glob', 'NotebookEdit']
    if tool_name in file_operation_tools:
        file_path = os.environ.get('CLAUDE_TOOL_FILE_PATH')
        if file_path:
            tool_info['file_path'] = file_path
            tool_info['working_directory'] = os.getcwd()
            
            # Add file operation metadata
            try:
                if os.path.exists(file_path):
                    stat_info = os.stat(file_path)
                    tool_info['file_size'] = stat_info.st_size
                    tool_info['file_modified'] = stat_info.st_mtime
                else:
                    tool_info['file_exists'] = False
            except OSError:
                tool_info['file_stat_error'] = True
                
        # Extract operation type from parameters
        if 'tool_params' in tool_info and isinstance(tool_info['tool_params'], dict):
            params = tool_info['tool_params']
            if 'old_string' in params and 'new_string' in params:
                tool_info['operation_type'] = 'edit'
            elif 'content' in params:
                tool_info['operation_type'] = 'write'
            elif tool_name == 'Read':
                tool_info['operation_type'] = 'read'
            elif tool_name in ['LS', 'Glob']:
                tool_info['operation_type'] = 'list'
    
    # Add tool category classification
    tool_categories = {
        'Read': 'file_read',
        'Write': 'file_write', 
        'Edit': 'file_edit',
        'MultiEdit': 'file_edit',
        'NotebookEdit': 'file_edit',
        'LS': 'file_list',
        'Glob': 'file_search',
        'Bash': 'system_command',
        'WebFetch': 'network',
        'WebSearch': 'network',
        'Grep': 'text_search'
    }
    
    tool_info['tool_category'] = tool_categories.get(tool_name, 'unknown')
    
    # Performance metrics for this extraction
    extraction_duration = (time.time() - extraction_start) * 1000
    tool_info['extraction_duration_ms'] = round(extraction_duration, 2)
    
    return tool_info


def capture_tool_use():
    """
    Main function to capture tool usage and store in conversation system.
    
    Extracts tool information from environment variables set by Claude Code
    during tool execution and stores the event using ConversationStorageBackend.
    """
    try:
        # Extract tool execution information
        tool_info = extract_tool_info()
        
        # Skip if no meaningful tool information
        if tool_info.get('tool_name') == 'unknown':
            logger.debug("Skipping tool capture - no tool name available")
            return
        
        # Initialize storage backend with correct path
        db_path = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "conversations.duckdb")
        storage = ConversationStorageBackend(db_path=db_path)
        
        # Get conversation context
        conversation_id = get_conversation_id()
        
        # Determine event type based on tool execution status
        if tool_info['success']:
            event_type = 'tool_success'
        else:
            event_type = 'tool_error'
        
        # Prepare event data
        event_data = {
            'tool_name': tool_info['tool_name'],
            'tool_params': tool_info.get('tool_params', {}),
            'execution_status': 'success' if tool_info['success'] else 'error',
            'exit_code': tool_info.get('exit_code', 0),
            'duration_ms': tool_info.get('duration_ms'),
            'capture_method': 'ToolUse_hook',
            'hook_version': '1.0.0'
        }
        
        # Add output information
        if 'stdout' in tool_info:
            event_data['stdout'] = tool_info['stdout']
        if 'stderr' in tool_info:
            event_data['stderr'] = tool_info['stderr']
        if 'error' in tool_info:
            event_data['error_message'] = tool_info['error']
        
        # Add file context for file operations
        if 'file_path' in tool_info:
            event_data['file_path'] = tool_info['file_path']
            event_data['working_directory'] = tool_info.get('working_directory')
        
        # Store the conversation event
        event_id = storage.store_conversation_event(
            conversation_id=conversation_id,
            agent_type='claude-code',
            event_type=event_type,
            event_data=event_data,
            issue_number=None,  # Will be populated by session manager later
            parent_event_id=None,
            embedding=None  # Will be generated by embedding service later
        )
        
        # If this was an error, also store it as a conversation error
        if not tool_info['success']:
            error_message = tool_info.get('error', 'Tool execution failed')
            if 'stderr' in tool_info and tool_info['stderr']:
                error_message = tool_info['stderr']
            
            error_context = {
                'tool_name': tool_info['tool_name'],
                'tool_params': tool_info.get('tool_params', {}),
                'exit_code': tool_info.get('exit_code', -1)
            }
            
            storage.store_conversation_error(
                conversation_id=conversation_id,
                agent_type='claude-code',
                error_type='tool_execution_error',
                error_message=error_message,
                error_context=error_context,
                resolution_attempted=None,
                resolution_success=False
            )
        
        # Log successful capture (debug level)
        logger.debug(f"Captured tool use: {tool_info['tool_name']} -> {event_id}")
        
        storage.close()
        
    except Exception as e:
        # Log error but don't disrupt Claude Code operation
        logger.error(f"Failed to capture tool use: {e}")
        sys.exit(0)  # Silent failure for hook system


if __name__ == '__main__':
    capture_tool_use()