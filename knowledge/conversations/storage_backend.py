"""
DuckDB storage backend for agent conversations.

Implements the database layer for storing conversation events, decisions, and errors
with vector embedding support for semantic search capabilities.
"""

import json
import uuid
import hashlib
from datetime import datetime
from typing import Dict, List, Any, Optional, Tuple
import logging
import duckdb

logger = logging.getLogger(__name__)


class ConversationStorageBackend:
    """
    DuckDB-based storage backend for agent conversations.
    
    Provides high-performance storage and retrieval of conversation events,
    decisions, and error patterns with vector embedding support.
    """
    
    def __init__(self, db_path: str = "knowledge/conversations.duckdb"):
        """
        Initialize conversation storage backend.
        
        Args:
            db_path: Path to DuckDB database file
        """
        self.db_path = db_path
        self.connection = None
        self._initialize_database()
    
    def _initialize_database(self):
        """Initialize database connection and create tables."""
        try:
            self.connection = duckdb.connect(self.db_path)
            
            # Install VSS extension for vector similarity search
            self.connection.execute("INSTALL vss;")
            self.connection.execute("LOAD vss;")
            
            # Create schema
            self._create_schema()
            
            logger.info(f"Conversation storage backend initialized: {self.db_path}")
            
        except Exception as e:
            logger.error(f"Failed to initialize conversation storage: {e}")
            raise
    
    def _create_schema(self):
        """Create database schema for conversation storage."""
        
        # Conversation events table
        self.connection.execute("""
            CREATE TABLE IF NOT EXISTS conversation_events (
                event_id VARCHAR PRIMARY KEY,
                conversation_id VARCHAR NOT NULL,
                agent_type VARCHAR(50) NOT NULL,
                issue_number INTEGER,
                timestamp TIMESTAMP NOT NULL,
                event_type VARCHAR(50) NOT NULL,
                event_data JSON NOT NULL,
                parent_event_id VARCHAR,
                embedding FLOAT[768],
                context_hash VARCHAR(64),
                created_at TIMESTAMP DEFAULT NOW()
            )
        """)
        
        # Agent decisions table
        self.connection.execute("""
            CREATE TABLE IF NOT EXISTS agent_decisions (
                decision_id VARCHAR PRIMARY KEY,
                conversation_id VARCHAR NOT NULL,
                agent_type VARCHAR(50) NOT NULL,
                decision_point TEXT NOT NULL,
                options_considered JSON NOT NULL,
                chosen_option TEXT NOT NULL,
                rationale TEXT,
                confidence_score FLOAT DEFAULT 0.5,
                outcome VARCHAR(50),
                outcome_timestamp TIMESTAMP,
                learning_value FLOAT DEFAULT 0.0,
                embedding FLOAT[768],
                created_at TIMESTAMP DEFAULT NOW()
            )
        """)
        
        # Conversation errors table
        self.connection.execute("""
            CREATE TABLE IF NOT EXISTS conversation_errors (
                error_id VARCHAR PRIMARY KEY,
                conversation_id VARCHAR NOT NULL,
                agent_type VARCHAR(50) NOT NULL,
                error_type VARCHAR(100) NOT NULL,
                error_message TEXT NOT NULL,
                error_context JSON,
                resolution_attempted TEXT,
                resolution_success BOOLEAN,
                pattern_signature VARCHAR(128),
                embedding FLOAT[768],
                created_at TIMESTAMP DEFAULT NOW()
            )
        """)
        
        # Conversation metadata table
        self.connection.execute("""
            CREATE TABLE IF NOT EXISTS conversation_metadata (
                conversation_id VARCHAR PRIMARY KEY,
                agent_type VARCHAR(50) NOT NULL,
                issue_number INTEGER,
                start_timestamp TIMESTAMP NOT NULL,
                end_timestamp TIMESTAMP,
                status VARCHAR(50) DEFAULT 'active',
                total_events INTEGER DEFAULT 0,
                success BOOLEAN,
                error_count INTEGER DEFAULT 0,
                decision_count INTEGER DEFAULT 0,
                context_summary TEXT,
                created_at TIMESTAMP DEFAULT NOW()
            )
        """)
        
        # Create indexes for performance
        self._create_indexes()
    
    def _create_indexes(self):
        """Create indexes for query performance."""
        indexes = [
            "CREATE INDEX IF NOT EXISTS idx_conversation_events_conversation_id ON conversation_events(conversation_id)",
            "CREATE INDEX IF NOT EXISTS idx_conversation_events_agent_type ON conversation_events(agent_type)",
            "CREATE INDEX IF NOT EXISTS idx_conversation_events_timestamp ON conversation_events(timestamp)",
            "CREATE INDEX IF NOT EXISTS idx_conversation_events_event_type ON conversation_events(event_type)",
            "CREATE INDEX IF NOT EXISTS idx_agent_decisions_conversation_id ON agent_decisions(conversation_id)",
            "CREATE INDEX IF NOT EXISTS idx_agent_decisions_outcome ON agent_decisions(outcome)",
            "CREATE INDEX IF NOT EXISTS idx_conversation_errors_pattern ON conversation_errors(pattern_signature)",
            "CREATE INDEX IF NOT EXISTS idx_conversation_errors_type ON conversation_errors(error_type)",
            "CREATE INDEX IF NOT EXISTS idx_conversation_metadata_agent_type ON conversation_metadata(agent_type)",
            "CREATE INDEX IF NOT EXISTS idx_conversation_metadata_issue ON conversation_metadata(issue_number)"
        ]
        
        for index_sql in indexes:
            try:
                self.connection.execute(index_sql)
            except Exception as e:
                logger.warning(f"Index creation failed: {e}")
    
    def store_conversation_event(self,
                                conversation_id: str,
                                agent_type: str,
                                event_type: str,
                                event_data: Dict[str, Any],
                                issue_number: Optional[int] = None,
                                parent_event_id: Optional[str] = None,
                                embedding: Optional[List[float]] = None) -> str:
        """
        Store a conversation event.
        
        Args:
            conversation_id: Unique conversation identifier
            agent_type: Type of agent (rif-implementer, etc.)
            event_type: Type of event (start, tool_use, decision, error, completion)
            event_data: Event data as dictionary
            issue_number: Associated GitHub issue number
            parent_event_id: Parent event for threading
            embedding: Vector embedding for semantic search
            
        Returns:
            Event ID of stored event
        """
        event_id = str(uuid.uuid4())
        timestamp = datetime.now()
        
        # Generate context hash for deduplication
        context_data = {
            'agent_type': agent_type,
            'event_type': event_type,
            'event_data': event_data
        }
        context_hash = hashlib.md5(json.dumps(context_data, sort_keys=True).encode()).hexdigest()
        
        try:
            self.connection.execute("""
                INSERT INTO conversation_events (
                    event_id, conversation_id, agent_type, issue_number,
                    timestamp, event_type, event_data, parent_event_id,
                    embedding, context_hash
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """, [
                event_id, conversation_id, agent_type, issue_number,
                timestamp, event_type, json.dumps(event_data), parent_event_id,
                embedding, context_hash
            ])
            
            # Update conversation metadata
            self._update_conversation_metadata(conversation_id, agent_type, issue_number)
            
            logger.debug(f"Stored conversation event: {event_id} ({event_type})")
            return event_id
            
        except Exception as e:
            logger.error(f"Failed to store conversation event: {e}")
            raise
    
    def store_agent_decision(self,
                           conversation_id: str,
                           agent_type: str,
                           decision_point: str,
                           options_considered: List[Dict[str, Any]],
                           chosen_option: str,
                           rationale: Optional[str] = None,
                           confidence_score: float = 0.5,
                           embedding: Optional[List[float]] = None) -> str:
        """
        Store an agent decision.
        
        Args:
            conversation_id: Conversation this decision belongs to
            agent_type: Type of agent making decision
            decision_point: Description of decision point
            options_considered: List of options that were considered
            chosen_option: The option that was chosen
            rationale: Explanation for the choice
            confidence_score: Confidence level (0.0 to 1.0)
            embedding: Vector embedding for semantic search
            
        Returns:
            Decision ID
        """
        decision_id = str(uuid.uuid4())
        
        try:
            self.connection.execute("""
                INSERT INTO agent_decisions (
                    decision_id, conversation_id, agent_type, decision_point,
                    options_considered, chosen_option, rationale, confidence_score,
                    embedding
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
            """, [
                decision_id, conversation_id, agent_type, decision_point,
                json.dumps(options_considered), chosen_option, rationale,
                confidence_score, embedding
            ])
            
            self._increment_conversation_counter(conversation_id, 'decision_count')
            
            logger.debug(f"Stored agent decision: {decision_id}")
            return decision_id
            
        except Exception as e:
            logger.error(f"Failed to store agent decision: {e}")
            raise
    
    def store_conversation_error(self,
                               conversation_id: str,
                               agent_type: str,
                               error_type: str,
                               error_message: str,
                               error_context: Optional[Dict[str, Any]] = None,
                               resolution_attempted: Optional[str] = None,
                               resolution_success: bool = False,
                               embedding: Optional[List[float]] = None) -> str:
        """
        Store a conversation error.
        
        Args:
            conversation_id: Conversation where error occurred
            agent_type: Type of agent that encountered error
            error_type: Category/type of error
            error_message: Error message or description
            error_context: Additional context about the error
            resolution_attempted: Description of attempted resolution
            resolution_success: Whether resolution was successful
            embedding: Vector embedding for semantic search
            
        Returns:
            Error ID
        """
        error_id = str(uuid.uuid4())
        
        # Generate pattern signature for grouping similar errors
        pattern_data = {
            'agent_type': agent_type,
            'error_type': error_type,
            'error_message_prefix': error_message[:100] if error_message else ''
        }
        pattern_signature = hashlib.md5(json.dumps(pattern_data, sort_keys=True).encode()).hexdigest()[:16]
        
        try:
            self.connection.execute("""
                INSERT INTO conversation_errors (
                    error_id, conversation_id, agent_type, error_type,
                    error_message, error_context, resolution_attempted,
                    resolution_success, pattern_signature, embedding
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """, [
                error_id, conversation_id, agent_type, error_type,
                error_message, json.dumps(error_context) if error_context else None,
                resolution_attempted, resolution_success, pattern_signature, embedding
            ])
            
            self._increment_conversation_counter(conversation_id, 'error_count')
            
            logger.debug(f"Stored conversation error: {error_id} (pattern: {pattern_signature})")
            return error_id
            
        except Exception as e:
            logger.error(f"Failed to store conversation error: {e}")
            raise
    
    def start_conversation(self,
                          agent_type: str,
                          issue_number: Optional[int] = None,
                          context_summary: Optional[str] = None) -> str:
        """
        Start a new conversation tracking session.
        
        Args:
            agent_type: Type of agent starting conversation
            issue_number: Associated GitHub issue number
            context_summary: Summary of conversation context
            
        Returns:
            Conversation ID
        """
        conversation_id = str(uuid.uuid4())
        
        try:
            self.connection.execute("""
                INSERT INTO conversation_metadata (
                    conversation_id, agent_type, issue_number,
                    start_timestamp, context_summary
                ) VALUES (?, ?, ?, ?, ?)
            """, [
                conversation_id, agent_type, issue_number,
                datetime.now(), context_summary
            ])
            
            logger.info(f"Started conversation: {conversation_id} ({agent_type})")
            return conversation_id
            
        except Exception as e:
            logger.error(f"Failed to start conversation: {e}")
            raise
    
    def end_conversation(self,
                        conversation_id: str,
                        success: bool = True,
                        context_summary: Optional[str] = None):
        """
        Mark a conversation as completed.
        
        Args:
            conversation_id: Conversation to end
            success: Whether conversation completed successfully
            context_summary: Final summary of conversation
        """
        try:
            update_data = [datetime.now(), 'completed', success]
            update_sql = "UPDATE conversation_metadata SET end_timestamp = ?, status = ?, success = ?"
            
            if context_summary:
                update_sql += ", context_summary = ?"
                update_data.append(context_summary)
            
            update_sql += " WHERE conversation_id = ?"
            update_data.append(conversation_id)
            
            self.connection.execute(update_sql, update_data)
            
            logger.info(f"Ended conversation: {conversation_id} (success: {success})")
            
        except Exception as e:
            logger.error(f"Failed to end conversation: {e}")
            raise
    
    def _update_conversation_metadata(self,
                                    conversation_id: str,
                                    agent_type: str,
                                    issue_number: Optional[int]):
        """Update or create conversation metadata."""
        try:
            # Try to increment event count
            result = self.connection.execute("""
                UPDATE conversation_metadata 
                SET total_events = total_events + 1
                WHERE conversation_id = ?
            """, [conversation_id])
            
            # If no rows affected, create metadata entry
            if result.rowcount == 0:
                self.connection.execute("""
                    INSERT INTO conversation_metadata (
                        conversation_id, agent_type, issue_number,
                        start_timestamp, total_events
                    ) VALUES (?, ?, ?, ?, 1)
                """, [
                    conversation_id, agent_type, issue_number, datetime.now()
                ])
                
        except Exception as e:
            logger.warning(f"Failed to update conversation metadata: {e}")
    
    def _increment_conversation_counter(self, conversation_id: str, counter_field: str):
        """Increment a counter field in conversation metadata."""
        try:
            self.connection.execute(f"""
                UPDATE conversation_metadata 
                SET {counter_field} = {counter_field} + 1
                WHERE conversation_id = ?
            """, [conversation_id])
        except Exception as e:
            logger.warning(f"Failed to increment {counter_field}: {e}")
    
    def get_conversation_events(self,
                              conversation_id: str,
                              event_type: Optional[str] = None,
                              limit: int = 100) -> List[Dict[str, Any]]:
        """
        Retrieve events for a conversation.
        
        Args:
            conversation_id: Conversation to retrieve
            event_type: Filter by event type
            limit: Maximum number of events
            
        Returns:
            List of conversation events
        """
        try:
            sql = """
                SELECT * FROM conversation_events 
                WHERE conversation_id = ?
            """
            params = [conversation_id]
            
            if event_type:
                sql += " AND event_type = ?"
                params.append(event_type)
            
            sql += " ORDER BY timestamp ASC LIMIT ?"
            params.append(limit)
            
            result = self.connection.execute(sql, params)
            events = []
            
            for row in result.fetchall():
                event = dict(zip([col[0] for col in result.description], row))
                # Parse JSON fields
                event['event_data'] = json.loads(event['event_data']) if event['event_data'] else {}
                events.append(event)
            
            return events
            
        except Exception as e:
            logger.error(f"Failed to retrieve conversation events: {e}")
            return []
    
    def get_agent_decisions(self,
                          conversation_id: Optional[str] = None,
                          agent_type: Optional[str] = None,
                          outcome: Optional[str] = None,
                          limit: int = 50) -> List[Dict[str, Any]]:
        """
        Retrieve agent decisions with filters.
        
        Args:
            conversation_id: Filter by conversation
            agent_type: Filter by agent type
            outcome: Filter by decision outcome
            limit: Maximum number of decisions
            
        Returns:
            List of agent decisions
        """
        try:
            sql = "SELECT * FROM agent_decisions WHERE 1=1"
            params = []
            
            if conversation_id:
                sql += " AND conversation_id = ?"
                params.append(conversation_id)
            
            if agent_type:
                sql += " AND agent_type = ?"
                params.append(agent_type)
            
            if outcome:
                sql += " AND outcome = ?"
                params.append(outcome)
            
            sql += " ORDER BY created_at DESC LIMIT ?"
            params.append(limit)
            
            result = self.connection.execute(sql, params)
            decisions = []
            
            for row in result.fetchall():
                decision = dict(zip([col[0] for col in result.description], row))
                # Parse JSON fields
                decision['options_considered'] = json.loads(decision['options_considered']) if decision['options_considered'] else []
                decisions.append(decision)
            
            return decisions
            
        except Exception as e:
            logger.error(f"Failed to retrieve agent decisions: {e}")
            return []
    
    def get_conversation_errors(self,
                              pattern_signature: Optional[str] = None,
                              agent_type: Optional[str] = None,
                              error_type: Optional[str] = None,
                              resolved: Optional[bool] = None,
                              limit: int = 50) -> List[Dict[str, Any]]:
        """
        Retrieve conversation errors with filters.
        
        Args:
            pattern_signature: Filter by error pattern
            agent_type: Filter by agent type
            error_type: Filter by error type
            resolved: Filter by resolution status
            limit: Maximum number of errors
            
        Returns:
            List of conversation errors
        """
        try:
            sql = "SELECT * FROM conversation_errors WHERE 1=1"
            params = []
            
            if pattern_signature:
                sql += " AND pattern_signature = ?"
                params.append(pattern_signature)
            
            if agent_type:
                sql += " AND agent_type = ?"
                params.append(agent_type)
            
            if error_type:
                sql += " AND error_type = ?"
                params.append(error_type)
            
            if resolved is not None:
                sql += " AND resolution_success = ?"
                params.append(resolved)
            
            sql += " ORDER BY created_at DESC LIMIT ?"
            params.append(limit)
            
            result = self.connection.execute(sql, params)
            errors = []
            
            for row in result.fetchall():
                error = dict(zip([col[0] for col in result.description], row))
                # Parse JSON fields
                error['error_context'] = json.loads(error['error_context']) if error['error_context'] else {}
                errors.append(error)
            
            return errors
            
        except Exception as e:
            logger.error(f"Failed to retrieve conversation errors: {e}")
            return []
    
    def get_storage_stats(self) -> Dict[str, Any]:
        """Get storage statistics."""
        try:
            stats = {}
            
            # Count records in each table
            tables = ['conversation_events', 'agent_decisions', 'conversation_errors', 'conversation_metadata']
            
            for table in tables:
                result = self.connection.execute(f"SELECT COUNT(*) FROM {table}")
                stats[f"{table}_count"] = result.fetchone()[0]
            
            # Get date ranges
            result = self.connection.execute("""
                SELECT 
                    MIN(created_at) as earliest,
                    MAX(created_at) as latest
                FROM conversation_events
            """)
            
            row = result.fetchone()
            if row and row[0]:
                stats['earliest_event'] = row[0]
                stats['latest_event'] = row[1]
            
            # Agent type distribution
            result = self.connection.execute("""
                SELECT agent_type, COUNT(*) as count
                FROM conversation_metadata
                GROUP BY agent_type
                ORDER BY count DESC
            """)
            
            stats['agent_distribution'] = dict(result.fetchall())
            
            return stats
            
        except Exception as e:
            logger.error(f"Failed to get storage stats: {e}")
            return {}
    
    def close(self):
        """Close database connection."""
        if self.connection:
            self.connection.close()
            logger.info("Conversation storage backend closed")