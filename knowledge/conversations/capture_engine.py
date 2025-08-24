"""
Conversation capture engine for RIF agents.

Provides automatic capture of agent interactions, decisions, and errors
through hooks and context managers that require minimal agent code changes.
"""

import json
import time
import threading
import traceback
from datetime import datetime
from typing import Dict, List, Any, Optional, Callable, Union
from contextlib import contextmanager
import logging

from .storage_backend import ConversationStorageBackend
from .embedding_generator import create_embedding_generator, ConversationEmbeddingGenerator
from .session_manager import ConversationSessionManager

logger = logging.getLogger(__name__)


class ConversationCaptureEngine:
    """
    Engine for capturing agent conversations and interactions.
    
    Provides context managers and hooks for automatic conversation capture
    with minimal impact on agent performance and code complexity.
    """
    
    def __init__(self, storage_backend: Optional[ConversationStorageBackend] = None, 
                 enable_embeddings: bool = True,
                 session_manager: Optional[ConversationSessionManager] = None):
        """
        Initialize conversation capture engine.
        
        Args:
            storage_backend: Storage backend (creates default if None)
            enable_embeddings: Whether to enable embedding generation
            session_manager: Session manager (creates default if None)
        """
        self.storage = storage_backend or ConversationStorageBackend()
        self.session_manager = session_manager or ConversationSessionManager(self.storage)
        self.active_conversations = {}
        self.capture_enabled = True
        self.embedding_generator = None  # Will be initialized if needed
        self._lock = threading.RLock()
        
        # Initialize embedding generator if requested
        if enable_embeddings:
            self._setup_embedding_generator()
        
        # Performance tracking
        self.capture_stats = {
            'events_captured': 0,
            'decisions_captured': 0,
            'errors_captured': 0,
            'conversations_started': 0,
            'conversations_completed': 0,
            'avg_capture_latency_ms': 0.0
        }
    
    def enable_capture(self, enabled: bool = True):
        """Enable or disable conversation capture."""
        self.capture_enabled = enabled
        logger.info(f"Conversation capture {'enabled' if enabled else 'disabled'}")
    
    def set_embedding_generator(self, generator_func: Callable[[str], List[float]]):
        """
        Set embedding generator function for semantic search.
        
        Args:
            generator_func: Function that takes text and returns embedding vector
        """
        self.embedding_generator = generator_func
        logger.info("Embedding generator configured for conversation capture")
    
    @contextmanager
    def conversation_session(self,
                            agent_type: str,
                            issue_number: Optional[int] = None,
                            context_summary: Optional[str] = None):
        """
        Context manager for automatic conversation tracking.
        
        Args:
            agent_type: Type of agent (rif-implementer, etc.)
            issue_number: Associated GitHub issue number
            context_summary: Summary of conversation context
            
        Usage:
            with capture_engine.conversation_session("rif-implementer", issue_number=34) as conv_id:
                # Agent operations are automatically captured
                pass
        """
        if not self.capture_enabled:
            yield None
            return
        
        conversation_id = None
        start_time = time.time()
        
        try:
            # Start conversation using session manager
            conversation_id = self.session_manager.start_session(
                agent_type=agent_type,
                issue_number=issue_number,
                context_summary=context_summary
            )
            
            with self._lock:
                self.active_conversations[conversation_id] = {
                    'agent_type': agent_type,
                    'issue_number': issue_number,
                    'start_time': start_time,
                    'event_count': 0
                }
                self.capture_stats['conversations_started'] += 1
            
            yield conversation_id
            
        except Exception as e:
            logger.error(f"Error in conversation session: {e}")
            if conversation_id:
                self.capture_error(
                    conversation_id=conversation_id,
                    error_type="conversation_session_error",
                    error_message=str(e),
                    error_context={'traceback': traceback.format_exc()}
                )
            raise
            
        finally:
            # End conversation using session manager
            if conversation_id:
                try:
                    success = True  # Could be determined by exception handling
                    self.session_manager.end_session(conversation_id, success=success)
                    
                    with self._lock:
                        if conversation_id in self.active_conversations:
                            del self.active_conversations[conversation_id]
                        self.capture_stats['conversations_completed'] += 1
                        
                except Exception as e:
                    logger.error(f"Error ending conversation {conversation_id}: {e}")
    
    def capture_event(self,
                     conversation_id: str,
                     event_type: str,
                     event_data: Dict[str, Any],
                     parent_event_id: Optional[str] = None,
                     generate_embedding: bool = False) -> str:
        """
        Capture a conversation event.
        
        Args:
            conversation_id: Active conversation ID
            event_type: Type of event (tool_use, decision, completion, etc.)
            event_data: Event-specific data
            parent_event_id: Parent event for threading
            generate_embedding: Whether to generate embedding for search
            
        Returns:
            Event ID
        """
        if not self.capture_enabled or not conversation_id:
            return ""
        
        capture_start = time.time()
        
        try:
            # Get conversation context
            with self._lock:
                conv_context = self.active_conversations.get(conversation_id, {})
            
            agent_type = conv_context.get('agent_type', 'unknown')
            issue_number = conv_context.get('issue_number')
            
            # Generate embedding if requested and generator available
            embedding = None
            if generate_embedding and self.embedding_generator:
                try:
                    # Create searchable text from event data
                    searchable_text = self._create_searchable_text(event_type, event_data)
                    embedding = self.embedding_generator(searchable_text)
                except Exception as e:
                    logger.warning(f"Failed to generate embedding: {e}")
            
            # Store event
            event_id = self.storage.store_conversation_event(
                conversation_id=conversation_id,
                agent_type=agent_type,
                event_type=event_type,
                event_data=event_data,
                issue_number=issue_number,
                parent_event_id=parent_event_id,
                embedding=embedding
            )
            
            # Update stats
            with self._lock:
                if conversation_id in self.active_conversations:
                    self.active_conversations[conversation_id]['event_count'] += 1
                
                self.capture_stats['events_captured'] += 1
                
                # Update average latency
                capture_latency = (time.time() - capture_start) * 1000
                current_avg = self.capture_stats['avg_capture_latency_ms']
                total_events = self.capture_stats['events_captured']
                self.capture_stats['avg_capture_latency_ms'] = (
                    (current_avg * (total_events - 1) + capture_latency) / total_events
                )
            
            return event_id
            
        except Exception as e:
            logger.error(f"Failed to capture event: {e}")
            return ""
    
    def capture_decision(self,
                        conversation_id: str,
                        decision_point: str,
                        options_considered: List[Union[str, Dict[str, Any]]],
                        chosen_option: str,
                        rationale: Optional[str] = None,
                        confidence_score: float = 0.5) -> str:
        """
        Capture an agent decision.
        
        Args:
            conversation_id: Active conversation ID
            decision_point: Description of the decision point
            options_considered: List of options that were considered
            chosen_option: The option that was chosen
            rationale: Explanation for the choice
            confidence_score: Confidence level (0.0 to 1.0)
            
        Returns:
            Decision ID
        """
        if not self.capture_enabled or not conversation_id:
            return ""
        
        try:
            # Get conversation context
            with self._lock:
                conv_context = self.active_conversations.get(conversation_id, {})
            
            agent_type = conv_context.get('agent_type', 'unknown')
            
            # Normalize options to dictionaries
            normalized_options = []
            for option in options_considered:
                if isinstance(option, str):
                    normalized_options.append({'option': option})
                elif isinstance(option, dict):
                    normalized_options.append(option)
                else:
                    normalized_options.append({'option': str(option)})
            
            # Generate embedding for decision
            embedding = None
            if self.embedding_generator:
                try:
                    decision_text = f"Decision: {decision_point}. Chosen: {chosen_option}. Rationale: {rationale or 'Not provided'}"
                    embedding = self.embedding_generator(decision_text)
                except Exception as e:
                    logger.warning(f"Failed to generate decision embedding: {e}")
            
            # Store decision
            decision_id = self.storage.store_agent_decision(
                conversation_id=conversation_id,
                agent_type=agent_type,
                decision_point=decision_point,
                options_considered=normalized_options,
                chosen_option=chosen_option,
                rationale=rationale,
                confidence_score=confidence_score,
                embedding=embedding
            )
            
            # Update stats
            with self._lock:
                self.capture_stats['decisions_captured'] += 1
            
            return decision_id
            
        except Exception as e:
            logger.error(f"Failed to capture decision: {e}")
            return ""
    
    def capture_error(self,
                     conversation_id: str,
                     error_type: str,
                     error_message: str,
                     error_context: Optional[Dict[str, Any]] = None,
                     resolution_attempted: Optional[str] = None,
                     resolution_success: bool = False) -> str:
        """
        Capture a conversation error.
        
        Args:
            conversation_id: Active conversation ID
            error_type: Category/type of error
            error_message: Error message or description
            error_context: Additional context about the error
            resolution_attempted: Description of attempted resolution
            resolution_success: Whether resolution was successful
            
        Returns:
            Error ID
        """
        if not self.capture_enabled or not conversation_id:
            return ""
        
        try:
            # Get conversation context
            with self._lock:
                conv_context = self.active_conversations.get(conversation_id, {})
            
            agent_type = conv_context.get('agent_type', 'unknown')
            
            # Generate embedding for error
            embedding = None
            if self.embedding_generator:
                try:
                    error_text = f"Error: {error_type}. Message: {error_message}. Resolution: {resolution_attempted or 'None attempted'}"
                    embedding = self.embedding_generator(error_text)
                except Exception as e:
                    logger.warning(f"Failed to generate error embedding: {e}")
            
            # Store error
            error_id = self.storage.store_conversation_error(
                conversation_id=conversation_id,
                agent_type=agent_type,
                error_type=error_type,
                error_message=error_message,
                error_context=error_context,
                resolution_attempted=resolution_attempted,
                resolution_success=resolution_success,
                embedding=embedding
            )
            
            # Update stats
            with self._lock:
                self.capture_stats['errors_captured'] += 1
            
            return error_id
            
        except Exception as e:
            logger.error(f"Failed to capture error: {e}")
            return ""
    
    def _create_searchable_text(self, event_type: str, event_data: Dict[str, Any]) -> str:
        """Create searchable text from event data."""
        text_parts = [f"Event: {event_type}"]
        
        # Extract key text fields from event data
        text_fields = ['description', 'summary', 'content', 'message', 'result', 'output']
        
        for field in text_fields:
            if field in event_data:
                value = event_data[field]
                if isinstance(value, str) and len(value.strip()) > 0:
                    text_parts.append(f"{field}: {value[:500]}")  # Limit length
        
        # Add other relevant fields
        if 'tool_name' in event_data:
            text_parts.append(f"Tool: {event_data['tool_name']}")
        
        if 'status' in event_data:
            text_parts.append(f"Status: {event_data['status']}")
        
        return ". ".join(text_parts)
    
    @contextmanager 
    def capture_tool_use(self,
                        conversation_id: str,
                        tool_name: str,
                        tool_params: Dict[str, Any],
                        parent_event_id: Optional[str] = None):
        """
        Context manager for capturing tool usage.
        
        Args:
            conversation_id: Active conversation ID
            tool_name: Name of tool being used
            tool_params: Parameters passed to tool
            parent_event_id: Parent event ID
            
        Usage:
            with capture_engine.capture_tool_use(conv_id, "Read", {"file_path": "/path"}) as tool_event_id:
                result = read_tool.execute(params)
                # Tool result is automatically captured
        """
        if not self.capture_enabled:
            yield None
            return
        
        tool_start_time = time.time()
        tool_event_id = None
        
        try:
            # Capture tool start event
            start_event_data = {
                'tool_name': tool_name,
                'tool_params': tool_params,
                'status': 'started',
                'start_time': datetime.now().isoformat()
            }
            
            tool_event_id = self.capture_event(
                conversation_id=conversation_id,
                event_type='tool_start',
                event_data=start_event_data,
                parent_event_id=parent_event_id,
                generate_embedding=True
            )
            
            yield tool_event_id
            
        except Exception as e:
            # Capture tool error
            error_event_data = {
                'tool_name': tool_name,
                'tool_params': tool_params,
                'status': 'error',
                'error': str(e),
                'duration_ms': (time.time() - tool_start_time) * 1000
            }
            
            self.capture_event(
                conversation_id=conversation_id,
                event_type='tool_error',
                event_data=error_event_data,
                parent_event_id=tool_event_id,
                generate_embedding=True
            )
            
            # Also capture as error for pattern detection
            self.capture_error(
                conversation_id=conversation_id,
                error_type='tool_execution_error',
                error_message=f"Tool '{tool_name}' failed: {str(e)}",
                error_context={
                    'tool_name': tool_name,
                    'tool_params': tool_params
                }
            )
            
            raise
            
        else:
            # Capture successful tool completion
            success_event_data = {
                'tool_name': tool_name,
                'tool_params': tool_params,
                'status': 'completed',
                'duration_ms': (time.time() - tool_start_time) * 1000
            }
            
            self.capture_event(
                conversation_id=conversation_id,
                event_type='tool_success',
                event_data=success_event_data,
                parent_event_id=tool_event_id,
                generate_embedding=True
            )
    
    def get_active_conversations(self) -> Dict[str, Dict[str, Any]]:
        """Get information about active conversations."""
        with self._lock:
            return self.active_conversations.copy()
    
    def get_capture_stats(self) -> Dict[str, Any]:
        """Get conversation capture statistics."""
        with self._lock:
            stats = self.capture_stats.copy()
        
        # Add active conversation count
        stats['active_conversations'] = len(self.active_conversations)
        
        # Add storage stats
        storage_stats = self.storage.get_storage_stats()
        stats['storage'] = storage_stats
        
        return stats
    
    def update_decision_outcome(self,
                              decision_id: str,
                              outcome: str,
                              learning_value: float = 0.0):
        """
        Update decision outcome for learning purposes.
        
        Args:
            decision_id: Decision ID to update
            outcome: Outcome (success, failure, partial, unknown)
            learning_value: Learning value derived from outcome (0.0 to 1.0)
        """
        try:
            self.storage.connection.execute("""
                UPDATE agent_decisions 
                SET outcome = ?, outcome_timestamp = ?, learning_value = ?
                WHERE decision_id = ?
            """, [outcome, datetime.now(), learning_value, decision_id])
            
            logger.debug(f"Updated decision outcome: {decision_id} -> {outcome}")
            
        except Exception as e:
            logger.error(f"Failed to update decision outcome: {e}")
    
    def close(self):
        """Close capture engine and storage backend."""
        if self.storage:
            self.storage.close()
        logger.info("Conversation capture engine closed")
    
    def _setup_embedding_generator(self):
        """Set up embedding generator with auto-training from existing data"""
        try:
            # Create embedding generator
            self._embedding_generator_instance = create_embedding_generator()
            
            # Auto-train if not already trained
            if not self._embedding_generator_instance.is_trained:
                logger.info("Auto-training embedding model from existing conversation data...")
                self._embedding_generator_instance.auto_train_from_storage(self.storage)
            
            # Set up generator function
            def embedding_function(text: str) -> Optional[List[float]]:
                return self._embedding_generator_instance.generate_embedding(text)
            
            self.set_embedding_generator(embedding_function)
            logger.info("Embedding generator set up successfully")
            
        except Exception as e:
            logger.warning(f"Failed to set up embedding generator: {e}. Continuing without embeddings.")
    
    def get_embedding_generator_stats(self) -> Optional[Dict[str, Any]]:
        """Get embedding generator statistics"""
        if hasattr(self, '_embedding_generator_instance'):
            return self._embedding_generator_instance.get_generation_stats()
        return None
    
    def get_session_manager_stats(self) -> Dict[str, Any]:
        """Get session manager statistics"""
        return self.session_manager.get_session_statistics()
    
    def pause_conversation(self, conversation_id: str, reason: Optional[str] = None) -> bool:
        """Pause an active conversation"""
        return self.session_manager.pause_session(conversation_id, reason)
    
    def resume_conversation(self, conversation_id: str) -> bool:
        """Resume a paused conversation"""
        return self.session_manager.resume_session(conversation_id)
    
    def get_conversation_metadata(self, conversation_id: str) -> Optional[Any]:
        """Get metadata for a conversation"""
        return self.session_manager.get_session_metadata(conversation_id)


# Convenience functions for use in agent code

def create_capture_engine(db_path: str = "knowledge/conversations.duckdb") -> ConversationCaptureEngine:
    """
    Factory function to create conversation capture engine.
    
    Args:
        db_path: Path to conversation database
        
    Returns:
        Configured capture engine
    """
    storage = ConversationStorageBackend(db_path)
    return ConversationCaptureEngine(storage)


# Global instance for easy access (can be overridden by agents)
_default_capture_engine = None

def get_default_capture_engine() -> ConversationCaptureEngine:
    """Get or create default capture engine."""
    global _default_capture_engine
    if _default_capture_engine is None:
        _default_capture_engine = create_capture_engine()
    return _default_capture_engine