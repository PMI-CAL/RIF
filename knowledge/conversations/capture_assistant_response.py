#!/usr/bin/env python3
"""
AssistantResponse Capture Hook for Claude Code.

Captures assistant responses from Claude Code and stores them in the
conversation system via ConversationStorageBackend.

This hook is triggered after Claude generates a response and captures:
- Response text and structure
- Response timing and performance metrics
- Tool usage summary within the response
- Response quality indicators (length, complexity)
- Context and reasoning patterns

Issue #45: Implement AssistantResponse capture hook (complexity:medium)
"""

import os
import sys
import json
import uuid
import time
import re
from datetime import datetime
from typing import Dict, Any, Optional, List
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
    
    Uses the same session file as other hooks to maintain
    conversation continuity.
    
    Returns:
        Conversation ID for current session
    """
    try:
        # Use same session file as other hooks
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


def analyze_response_structure(response_text: str) -> Dict[str, Any]:
    """
    Analyze the structure and characteristics of the assistant response.
    
    Args:
        response_text: The full text of the assistant's response
        
    Returns:
        Dictionary containing response analysis
    """
    analysis = {
        'response_length': len(response_text),
        'word_count': len(response_text.split()),
        'paragraph_count': len([p for p in response_text.split('\n\n') if p.strip()]),
        'line_count': len(response_text.split('\n')),
    }
    
    # Detect tool calls in the response
    tool_call_pattern = r'<function_calls>'
    tool_calls = re.findall(tool_call_pattern, response_text)
    analysis['tool_calls_count'] = len(tool_calls)
    
    # Extract tool names if present
    tool_name_pattern = r'<invoke name="([^"]+)">'
    tool_names = re.findall(tool_name_pattern, response_text)
    analysis['tools_used'] = list(set(tool_names))  # Unique tools
    
    # Detect code blocks
    code_block_pattern = r'```[\w]*\n.*?\n```'
    code_blocks = re.findall(code_block_pattern, response_text, re.DOTALL)
    analysis['code_blocks_count'] = len(code_blocks)
    
    # Detect file paths and references
    file_path_pattern = r'[/\\][\w.-]+(?:[/\\][\w.-]+)*\.[\w]+|[/\\][\w.-]+(?:[/\\][\w.-]+)*[/\\]'
    file_paths = re.findall(file_path_pattern, response_text)
    analysis['file_references'] = list(set(file_paths))
    
    # Detect reasoning patterns
    reasoning_indicators = [
        'Let me', 'First, I', 'Then I', 'Next, I', 'Now I',
        'I need to', 'I should', 'I will', 'I can',
        'Based on', 'Looking at', 'Given that', 'Since',
        'Therefore', 'However', 'Additionally', 'Moreover'
    ]
    
    reasoning_count = 0
    for indicator in reasoning_indicators:
        reasoning_count += len(re.findall(re.escape(indicator), response_text, re.IGNORECASE))
    
    analysis['reasoning_indicators_count'] = reasoning_count
    
    # Detect question or explanation patterns
    questions = len(re.findall(r'\?', response_text))
    explanations = len(re.findall(r'\b(because|since|due to|as a result)\b', response_text, re.IGNORECASE))
    
    analysis['questions_count'] = questions
    analysis['explanations_count'] = explanations
    
    # Response type classification
    response_types = []
    
    if analysis['tool_calls_count'] > 0:
        response_types.append('action')
    if analysis['code_blocks_count'] > 0:
        response_types.append('code')
    if analysis['explanations_count'] > 2:
        response_types.append('explanation')
    if questions > 0:
        response_types.append('question')
    if analysis['file_references']:
        response_types.append('file_operation')
    
    if not response_types:
        response_types.append('general')
    
    analysis['response_types'] = response_types
    
    return analysis


def extract_response_context() -> Dict[str, Any]:
    """
    Extract context information about the assistant response.
    
    Returns:
        Dictionary containing response context
    """
    context = {
        'timestamp': datetime.now().isoformat(),
        'session_id': os.environ.get('CLAUDE_SESSION_ID', 'unknown'),
        'working_directory': os.getcwd(),
    }
    
    # Add Claude Code specific context
    claude_context = {}
    for key, value in os.environ.items():
        if key.startswith('CLAUDE_'):
            claude_context[key.lower()] = value
    
    if claude_context:
        context['claude_context'] = claude_context
    
    # Get response timing if available
    response_start_time = os.environ.get('CLAUDE_RESPONSE_START_TIME')
    response_end_time = os.environ.get('CLAUDE_RESPONSE_END_TIME')
    if response_start_time and response_end_time:
        try:
            duration_ms = (float(response_end_time) - float(response_start_time)) * 1000
            context['response_duration_ms'] = duration_ms
        except (ValueError, TypeError):
            pass
    
    # Get model information if available
    model_name = os.environ.get('CLAUDE_MODEL_NAME', 'unknown')
    context['model_name'] = model_name
    
    return context


def capture_assistant_response():
    """
    Main function to capture assistant response and store in conversation system.
    
    Reads the assistant response from stdin, analyzes its structure and content,
    and stores the event using ConversationStorageBackend.
    """
    try:
        # Read assistant response from stdin (Claude Code hook mechanism)
        response_text = sys.stdin.read().strip()
        
        # Skip empty responses
        if not response_text:
            logger.debug("Skipping empty assistant response")
            return
        
        # Initialize storage backend with correct path
        db_path = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "conversations.duckdb")
        storage = ConversationStorageBackend(db_path=db_path)
        
        # Get conversation context
        conversation_id = get_conversation_id()
        response_context = extract_response_context()
        
        # Analyze response structure and content
        response_analysis = analyze_response_structure(response_text)
        
        # Prepare event data
        event_data = {
            'response_text': response_text[:5000],  # Limit stored text size
            'response_length': len(response_text),
            'response_hash': str(hash(response_text)),  # For duplicate detection
            'response_context': response_context,
            'response_analysis': response_analysis,
            'capture_method': 'AssistantResponse_hook',
            'hook_version': '1.0.0'
        }
        
        # Add response timing if available
        if 'response_duration_ms' in response_context:
            event_data['response_duration_ms'] = response_context['response_duration_ms']
        
        # Store the conversation event
        event_id = storage.store_conversation_event(
            conversation_id=conversation_id,
            agent_type='claude-code',
            event_type='assistant_response',
            event_data=event_data,
            issue_number=None,  # Will be populated by session manager later
            parent_event_id=None,
            embedding=None  # Will be generated by embedding service later
        )
        
        # If the response contains tool usage, extract decision information
        if response_analysis.get('tools_used'):
            decision_point = f"Tool selection for response"
            options_considered = [{"option": f"Use {tool}"} for tool in response_analysis['tools_used']]
            chosen_option = f"Use tools: {', '.join(response_analysis['tools_used'])}"
            
            rationale = "Selected based on task requirements"
            if response_analysis['tool_calls_count'] > 1:
                rationale = "Multiple tools selected for complex task"
            
            # Estimate confidence based on response characteristics
            confidence_score = 0.8  # Default
            if response_analysis['reasoning_indicators_count'] > 3:
                confidence_score = 0.9  # High reasoning indicates confidence
            elif response_analysis['questions_count'] > 0:
                confidence_score = 0.6  # Questions indicate uncertainty
            
            storage.store_agent_decision(
                conversation_id=conversation_id,
                agent_type='claude-code',
                decision_point=decision_point,
                options_considered=options_considered,
                chosen_option=chosen_option,
                rationale=rationale,
                confidence_score=confidence_score
            )
        
        # Log successful capture (debug level)
        logger.debug(f"Captured assistant response: {event_id} (length: {len(response_text)}, tools: {response_analysis.get('tools_used', [])})")
        
        storage.close()
        
    except Exception as e:
        # Log error but don't disrupt Claude Code operation
        logger.error(f"Failed to capture assistant response: {e}")
        sys.exit(0)  # Silent failure for hook system


if __name__ == '__main__':
    capture_assistant_response()