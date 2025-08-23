"""
Query engine for agent conversation analysis and retrieval.

Provides natural language and structured query capabilities for searching
through agent conversations, decisions, and error patterns.
"""

import json
from datetime import datetime, timedelta
from typing import Dict, List, Any, Optional, Tuple
import logging

from .storage_backend import ConversationStorageBackend
from ..context.optimizer import ContextOptimizer

logger = logging.getLogger(__name__)


class ConversationQueryEngine:
    """
    Query engine for agent conversation analysis and retrieval.
    
    Supports semantic search, pattern analysis, and structured queries
    across all stored agent conversations.
    """
    
    def __init__(self, 
                 storage_backend: Optional[ConversationStorageBackend] = None,
                 context_optimizer: Optional[ContextOptimizer] = None):
        """
        Initialize conversation query engine.
        
        Args:
            storage_backend: Storage backend (creates default if None)
            context_optimizer: Context optimizer for result optimization
        """
        self.storage = storage_backend or ConversationStorageBackend()
        self.context_optimizer = context_optimizer or ContextOptimizer()
        self.embedding_generator = None
        
        # Query performance tracking
        self.query_stats = {
            'queries_executed': 0,
            'avg_query_time_ms': 0.0,
            'semantic_searches': 0,
            'structured_queries': 0
        }
    
    def set_embedding_generator(self, generator_func):
        """Set embedding generator for semantic search."""
        self.embedding_generator = generator_func
        logger.info("Embedding generator configured for conversation queries")
    
    def search_conversations(self,
                           query: str,
                           agent_type: Optional[str] = None,
                           issue_number: Optional[int] = None,
                           event_type: Optional[str] = None,
                           start_date: Optional[datetime] = None,
                           end_date: Optional[datetime] = None,
                           limit: int = 10,
                           optimize_for_agent: str = 'default') -> List[Dict[str, Any]]:
        """
        Search conversations using natural language query.
        
        Args:
            query: Natural language search query
            agent_type: Filter by agent type
            issue_number: Filter by issue number
            event_type: Filter by event type
            start_date: Filter by start date
            end_date: Filter by end date
            limit: Maximum number of results
            optimize_for_agent: Optimize results for specific agent type
            
        Returns:
            List of matching conversation events with relevance scores
        """
        query_start = datetime.now()
        
        try:
            # Build structured query
            filters = {}
            if agent_type:
                filters['agent_type'] = agent_type
            if issue_number:
                filters['issue_number'] = issue_number
            if event_type:
                filters['event_type'] = event_type
            
            # Execute search
            if self.embedding_generator:
                results = self._semantic_search(
                    query=query,
                    filters=filters,
                    start_date=start_date,
                    end_date=end_date,
                    limit=limit * 2  # Get more for optimization
                )
                self.query_stats['semantic_searches'] += 1
            else:
                results = self._text_search(
                    query=query,
                    filters=filters,
                    start_date=start_date,
                    end_date=end_date,
                    limit=limit * 2
                )
                self.query_stats['structured_queries'] += 1
            
            # Optimize results for agent consumption
            if results and self.context_optimizer:
                optimization_result = self.context_optimizer.optimize_for_agent(
                    results=results,
                    query=query,
                    agent_type=optimize_for_agent,
                    context={
                        'search_type': 'conversations',
                        'filters': filters
                    },
                    min_results=min(limit, len(results))
                )
                optimized_results = optimization_result['optimized_results'][:limit]
            else:
                optimized_results = results[:limit]
            
            # Update performance stats
            self._update_query_stats(query_start)
            
            return optimized_results
            
        except Exception as e:
            logger.error(f"Conversation search failed: {e}")
            return []
    
    def _semantic_search(self,
                        query: str,
                        filters: Dict[str, Any],
                        start_date: Optional[datetime],
                        end_date: Optional[datetime],
                        limit: int) -> List[Dict[str, Any]]:
        """Perform semantic search using vector embeddings."""
        
        # Generate query embedding
        query_embedding = self.embedding_generator(query)
        
        # Build SQL for vector similarity search
        sql = """
            SELECT 
                e.*,
                array_cosine_similarity(e.embedding, ?::FLOAT[768]) as similarity_score
            FROM conversation_events e
            WHERE e.embedding IS NOT NULL
        """
        
        params = [query_embedding]
        
        # Add filters
        if filters.get('agent_type'):
            sql += " AND e.agent_type = ?"
            params.append(filters['agent_type'])
        
        if filters.get('issue_number'):
            sql += " AND e.issue_number = ?"
            params.append(filters['issue_number'])
        
        if filters.get('event_type'):
            sql += " AND e.event_type = ?"
            params.append(filters['event_type'])
        
        if start_date:
            sql += " AND e.timestamp >= ?"
            params.append(start_date)
        
        if end_date:
            sql += " AND e.timestamp <= ?"
            params.append(end_date)
        
        # Order by similarity and limit
        sql += " ORDER BY similarity_score DESC LIMIT ?"
        params.append(limit)
        
        try:
            result = self.storage.connection.execute(sql, params)
            events = []
            
            for row in result.fetchall():
                event = dict(zip([col[0] for col in result.description], row))
                # Parse JSON fields
                event['event_data'] = json.loads(event['event_data']) if event['event_data'] else {}
                # Add distance for compatibility with context optimizer
                event['distance'] = 1.0 - event['similarity_score']  # Convert similarity to distance
                events.append(event)
            
            return events
            
        except Exception as e:
            logger.error(f"Semantic search failed: {e}")
            return []
    
    def _text_search(self,
                    query: str,
                    filters: Dict[str, Any],
                    start_date: Optional[datetime],
                    end_date: Optional[datetime],
                    limit: int) -> List[Dict[str, Any]]:
        """Perform text-based search when embeddings not available."""
        
        # Extract search terms
        search_terms = [term.strip().lower() for term in query.split() if len(term.strip()) > 2]
        
        if not search_terms:
            return []
        
        # Build text search SQL - join with metadata to search context_summary too
        sql = """
            SELECT 
                e.*,
                0.5 as similarity_score
            FROM conversation_events e
            JOIN conversation_metadata m ON e.conversation_id = m.conversation_id
            WHERE (
                LOWER(CAST(e.event_data AS VARCHAR)) LIKE ?
                OR LOWER(COALESCE(m.context_summary, '')) LIKE ?
        """
        
        # Create LIKE patterns for each search term
        full_query_pattern = f"%{' '.join(search_terms)}%"
        params = [full_query_pattern, full_query_pattern]
        
        for term in search_terms:
            sql += f" OR LOWER(CAST(e.event_data AS VARCHAR)) LIKE ?"
            sql += f" OR LOWER(COALESCE(m.context_summary, '')) LIKE ?"
            params.extend([f"%{term}%", f"%{term}%"])
        
        sql += ")"
        
        # Add filters
        if filters.get('agent_type'):
            sql += " AND e.agent_type = ?"
            params.append(filters['agent_type'])
        
        if filters.get('issue_number'):
            sql += " AND m.issue_number = ?"
            params.append(filters['issue_number'])
        
        if filters.get('event_type'):
            sql += " AND e.event_type = ?"
            params.append(filters['event_type'])
        
        if start_date:
            sql += " AND e.timestamp >= ?"
            params.append(start_date)
        
        if end_date:
            sql += " AND e.timestamp <= ?"
            params.append(end_date)
        
        # Order by timestamp and limit
        sql += " ORDER BY e.timestamp DESC LIMIT ?"
        params.append(limit)
        
        try:
            result = self.storage.connection.execute(sql, params)
            events = []
            
            for row in result.fetchall():
                event = dict(zip([col[0] for col in result.description], row))
                # Parse JSON fields
                event['event_data'] = json.loads(event['event_data']) if event['event_data'] else {}
                # Add distance for compatibility
                event['distance'] = 0.5  # Neutral distance for text search
                events.append(event)
            
            return events
            
        except Exception as e:
            logger.error(f"Text search failed: {e}")
            return []
    
    def find_similar_decisions(self,
                             decision_description: str,
                             agent_type: Optional[str] = None,
                             outcome_filter: Optional[str] = None,
                             limit: int = 10) -> List[Dict[str, Any]]:
        """
        Find decisions similar to a given description.
        
        Args:
            decision_description: Description to match against
            agent_type: Filter by agent type
            outcome_filter: Filter by decision outcome
            limit: Maximum number of results
            
        Returns:
            List of similar decisions with similarity scores
        """
        try:
            if self.embedding_generator:
                # Semantic search using embeddings
                query_embedding = self.embedding_generator(decision_description)
                
                sql = """
                    SELECT 
                        d.*,
                        array_cosine_similarity(d.embedding, ?::FLOAT[768]) as similarity_score
                    FROM agent_decisions d
                    WHERE d.embedding IS NOT NULL
                """
                
                params = [query_embedding]
                
                if agent_type:
                    sql += " AND d.agent_type = ?"
                    params.append(agent_type)
                
                if outcome_filter:
                    sql += " AND d.outcome = ?"
                    params.append(outcome_filter)
                
                sql += " ORDER BY similarity_score DESC LIMIT ?"
                params.append(limit)
                
            else:
                # Text-based search
                search_terms = decision_description.lower()
                
                sql = """
                    SELECT 
                        d.*,
                        0.5 as similarity_score
                    FROM agent_decisions d
                    WHERE LOWER(d.decision_point) LIKE ? 
                       OR LOWER(d.chosen_option) LIKE ?
                       OR LOWER(COALESCE(d.rationale, '')) LIKE ?
                """
                
                params = [f"%{search_terms}%", f"%{search_terms}%", f"%{search_terms}%"]
                
                if agent_type:
                    sql += " AND d.agent_type = ?"
                    params.append(agent_type)
                
                if outcome_filter:
                    sql += " AND d.outcome = ?"
                    params.append(outcome_filter)
                
                sql += " ORDER BY d.created_at DESC LIMIT ?"
                params.append(limit)
            
            result = self.storage.connection.execute(sql, params)
            decisions = []
            
            for row in result.fetchall():
                decision = dict(zip([col[0] for col in result.description], row))
                # Parse JSON fields
                decision['options_considered'] = json.loads(decision['options_considered']) if decision['options_considered'] else []
                decisions.append(decision)
            
            return decisions
            
        except Exception as e:
            logger.error(f"Similar decisions search failed: {e}")
            return []
    
    def analyze_error_patterns(self,
                             error_type: Optional[str] = None,
                             agent_type: Optional[str] = None,
                             days_back: int = 30,
                             min_occurrences: int = 2) -> List[Dict[str, Any]]:
        """
        Analyze error patterns and trends.
        
        Args:
            error_type: Filter by error type
            agent_type: Filter by agent type
            days_back: Number of days to look back
            min_occurrences: Minimum occurrences to include pattern
            
        Returns:
            List of error patterns with occurrence counts and trends
        """
        try:
            start_date = datetime.now() - timedelta(days=days_back)
            
            sql = """
                SELECT 
                    pattern_signature,
                    error_type,
                    agent_type,
                    COUNT(*) as occurrence_count,
                    COUNT(CASE WHEN resolution_success = true THEN 1 END) as resolved_count,
                    MIN(created_at) as first_occurrence,
                    MAX(created_at) as last_occurrence,
                    AVG(CASE WHEN resolution_success = true THEN 1.0 ELSE 0.0 END) as resolution_rate
                FROM conversation_errors
                WHERE created_at >= ?
            """
            
            params = [start_date]
            
            if error_type:
                sql += " AND error_type = ?"
                params.append(error_type)
            
            if agent_type:
                sql += " AND agent_type = ?"
                params.append(agent_type)
            
            sql += """
                GROUP BY pattern_signature, error_type, agent_type
                HAVING COUNT(*) >= ?
                ORDER BY occurrence_count DESC, last_occurrence DESC
            """
            params.append(min_occurrences)
            
            result = self.storage.connection.execute(sql, params)
            patterns = []
            
            for row in result.fetchall():
                pattern = dict(zip([col[0] for col in result.description], row))
                patterns.append(pattern)
            
            return patterns
            
        except Exception as e:
            logger.error(f"Error pattern analysis failed: {e}")
            return []
    
    def get_agent_performance_summary(self,
                                    agent_type: str,
                                    days_back: int = 7) -> Dict[str, Any]:
        """
        Get performance summary for a specific agent.
        
        Args:
            agent_type: Agent type to analyze
            days_back: Number of days to analyze
            
        Returns:
            Performance summary statistics
        """
        try:
            start_date = datetime.now() - timedelta(days=days_back)
            
            # Get conversation statistics
            conv_result = self.storage.connection.execute("""
                SELECT 
                    COUNT(*) as total_conversations,
                    COUNT(CASE WHEN success = true THEN 1 END) as successful_conversations,
                    AVG(total_events) as avg_events_per_conversation,
                    AVG(decision_count) as avg_decisions_per_conversation,
                    AVG(error_count) as avg_errors_per_conversation
                FROM conversation_metadata
                WHERE agent_type = ? AND start_timestamp >= ?
            """, [agent_type, start_date])
            
            conv_stats = dict(zip([col[0] for col in conv_result.description], conv_result.fetchone()))
            
            # Get decision outcomes
            decision_result = self.storage.connection.execute("""
                SELECT 
                    outcome,
                    COUNT(*) as count,
                    AVG(confidence_score) as avg_confidence
                FROM agent_decisions
                WHERE agent_type = ? AND created_at >= ?
                GROUP BY outcome
            """, [agent_type, start_date])
            
            decision_outcomes = {}
            for row in decision_result.fetchall():
                outcome, count, avg_confidence = row
                decision_outcomes[outcome or 'unknown'] = {
                    'count': count,
                    'avg_confidence': avg_confidence
                }
            
            # Get error summary
            error_result = self.storage.connection.execute("""
                SELECT 
                    COUNT(*) as total_errors,
                    COUNT(CASE WHEN resolution_success = true THEN 1 END) as resolved_errors,
                    COUNT(DISTINCT pattern_signature) as unique_error_patterns
                FROM conversation_errors
                WHERE agent_type = ? AND created_at >= ?
            """, [agent_type, start_date])
            
            error_stats = dict(zip([col[0] for col in error_result.description], error_result.fetchone()))
            
            # Calculate derived metrics
            summary = {
                'agent_type': agent_type,
                'analysis_period_days': days_back,
                'conversation_stats': conv_stats,
                'decision_outcomes': decision_outcomes,
                'error_stats': error_stats,
                'derived_metrics': {
                    'success_rate': conv_stats['successful_conversations'] / max(conv_stats['total_conversations'], 1),
                    'error_resolution_rate': error_stats['resolved_errors'] / max(error_stats['total_errors'], 1),
                    'efficiency_score': self._calculate_efficiency_score(conv_stats, error_stats)
                }
            }
            
            return summary
            
        except Exception as e:
            logger.error(f"Performance summary failed: {e}")
            return {}
    
    def _calculate_efficiency_score(self, conv_stats: Dict[str, Any], error_stats: Dict[str, Any]) -> float:
        """Calculate efficiency score based on conversation and error stats."""
        try:
            # Base score from success rate
            success_rate = conv_stats['successful_conversations'] / max(conv_stats['total_conversations'], 1)
            base_score = success_rate * 100
            
            # Penalty for errors
            total_conversations = conv_stats['total_conversations']
            if total_conversations > 0:
                error_penalty = (error_stats['total_errors'] / total_conversations) * 20
                base_score = max(0, base_score - error_penalty)
            
            # Bonus for error resolution
            if error_stats['total_errors'] > 0:
                resolution_bonus = (error_stats['resolved_errors'] / error_stats['total_errors']) * 10
                base_score += resolution_bonus
            
            return min(100, max(0, base_score))
            
        except Exception:
            return 50.0  # Neutral score on calculation error
    
    def _update_query_stats(self, query_start: datetime):
        """Update query performance statistics."""
        query_time = (datetime.now() - query_start).total_seconds() * 1000
        
        self.query_stats['queries_executed'] += 1
        current_avg = self.query_stats['avg_query_time_ms']
        total_queries = self.query_stats['queries_executed']
        
        self.query_stats['avg_query_time_ms'] = (
            (current_avg * (total_queries - 1) + query_time) / total_queries
        )
    
    def get_query_stats(self) -> Dict[str, Any]:
        """Get query engine statistics."""
        stats = self.query_stats.copy()
        stats['embedding_enabled'] = self.embedding_generator is not None
        return stats
    
    def export_conversation(self,
                          conversation_id: str,
                          format_type: str = 'json') -> Optional[Dict[str, Any]]:
        """
        Export a complete conversation with all events, decisions, and errors.
        
        Args:
            conversation_id: Conversation to export
            format_type: Export format ('json', 'markdown')
            
        Returns:
            Exported conversation data
        """
        try:
            # Get conversation metadata
            metadata_result = self.storage.connection.execute("""
                SELECT * FROM conversation_metadata WHERE conversation_id = ?
            """, [conversation_id])
            
            metadata_row = metadata_result.fetchone()
            if not metadata_row:
                return None
            
            metadata = dict(zip([col[0] for col in metadata_result.description], metadata_row))
            
            # Get all events
            events = self.storage.get_conversation_events(conversation_id, limit=1000)
            
            # Get decisions
            decisions = self.storage.get_agent_decisions(conversation_id=conversation_id)
            
            # Get errors
            errors = self.storage.get_conversation_errors(agent_type=metadata['agent_type'])
            # Filter errors by conversation_id
            conversation_errors = [e for e in errors if e['conversation_id'] == conversation_id]
            
            # Compile export data
            export_data = {
                'conversation_id': conversation_id,
                'metadata': metadata,
                'events': events,
                'decisions': decisions,
                'errors': conversation_errors,
                'export_timestamp': datetime.now().isoformat(),
                'format': format_type
            }
            
            if format_type == 'markdown':
                export_data['markdown'] = self._format_conversation_as_markdown(export_data)
            
            return export_data
            
        except Exception as e:
            logger.error(f"Conversation export failed: {e}")
            return None
    
    def _format_conversation_as_markdown(self, conversation_data: Dict[str, Any]) -> str:
        """Format conversation data as markdown."""
        metadata = conversation_data['metadata']
        events = conversation_data['events']
        decisions = conversation_data['decisions']
        errors = conversation_data['errors']
        
        md_lines = [
            f"# Conversation {conversation_data['conversation_id']}",
            f"",
            f"**Agent**: {metadata['agent_type']}",
            f"**Issue**: #{metadata['issue_number']}" if metadata['issue_number'] else "",
            f"**Started**: {metadata['start_timestamp']}",
            f"**Status**: {metadata['status']}",
            f"**Total Events**: {metadata['total_events']}",
            f"",
            f"## Events ({len(events)})",
            f""
        ]
        
        for event in events:
            md_lines.extend([
                f"### {event['event_type'].title()} - {event['timestamp']}",
                f"```json",
                json.dumps(event['event_data'], indent=2),
                f"```",
                f""
            ])
        
        if decisions:
            md_lines.extend([
                f"## Decisions ({len(decisions)})",
                f""
            ])
            
            for decision in decisions:
                md_lines.extend([
                    f"### {decision['decision_point']}",
                    f"**Chosen**: {decision['chosen_option']}",
                    f"**Confidence**: {decision['confidence_score']}",
                    f"**Rationale**: {decision['rationale'] or 'Not provided'}",
                    f"**Outcome**: {decision['outcome'] or 'Unknown'}",
                    f""
                ])
        
        if errors:
            md_lines.extend([
                f"## Errors ({len(errors)})",
                f""
            ])
            
            for error in errors:
                md_lines.extend([
                    f"### {error['error_type']}: {error['error_message']}",
                    f"**Resolution Attempted**: {error['resolution_attempted'] or 'None'}",
                    f"**Resolved**: {'Yes' if error['resolution_success'] else 'No'}",
                    f""
                ])
        
        return "\n".join(line for line in md_lines if line is not None)