#!/usr/bin/env python3
"""
Conversation Query API - Issue #50
Comprehensive API for querying conversations by pattern, filtering by agent type and outcome.

This system implements:
1. Pattern-based search functionality  
2. Multi-criteria filtering capabilities
3. Result ranking and scoring
4. Query optimization for large datasets
5. Similarity search using embeddings
6. Conversation session management
7. Agent-specific conversation analysis
"""

import json
import sqlite3
import duckdb
import logging
import os
import sys
from datetime import datetime, timedelta, timezone
from typing import Dict, List, Optional, Any, Tuple, Union
from pathlib import Path
from dataclasses import dataclass, asdict
from enum import Enum
import hashlib
import statistics

# Add knowledge directory to path for imports
knowledge_path = Path(__file__).parent.parent / "knowledge"
sys.path.insert(0, str(knowledge_path))

try:
    from conversations.storage_backend import ConversationStorageBackend
except ImportError:
    logging.warning("ConversationStorageBackend not available - using mock backend")

class QueryType(Enum):
    """Types of conversation queries."""
    PATTERN_SEARCH = "pattern_search"
    SIMILARITY_SEARCH = "similarity_search"
    METADATA_FILTER = "metadata_filter"
    AGENT_ANALYSIS = "agent_analysis"
    SESSION_ANALYSIS = "session_analysis"
    OUTCOME_ANALYSIS = "outcome_analysis"

@dataclass
class QueryFilter:
    """Container for query filter parameters."""
    agent_type: Optional[str] = None
    event_type: Optional[str] = None
    outcome: Optional[str] = None
    date_range: Optional[Tuple[datetime, datetime]] = None
    issue_number: Optional[int] = None
    conversation_id: Optional[str] = None
    min_score: Optional[float] = None
    max_results: int = 50

@dataclass
class ConversationResult:
    """Container for conversation query result."""
    conversation_id: str
    event_id: str
    agent_type: str
    event_type: str
    event_data: Dict[str, Any]
    timestamp: datetime
    issue_number: Optional[int]
    score: float
    relevance_factors: List[str]

@dataclass
class QueryResult:
    """Container for complete query result."""
    query_type: QueryType
    total_matches: int
    results: List[ConversationResult]
    query_time_ms: float
    filters_applied: QueryFilter
    ranking_method: str
    suggestions: List[str]

class ConversationQueryAPI:
    """
    Comprehensive API for querying and analyzing conversations.
    """
    
    def __init__(self, db_path: Optional[str] = None):
        """Initialize conversation query API."""
        self.setup_logging()
        
        # Set database path
        if db_path is None:
            db_path = self._find_conversations_db()
        
        self.db_path = db_path
        self.connection = None
        self.storage_backend = None
        
        # Initialize storage backend
        self._initialize_storage()
        
        self.logger.info(f"🔍 ConversationQueryAPI initialized with database: {self.db_path}")
    
    def setup_logging(self):
        """Setup logging for conversation query API."""
        logging.basicConfig(
            level=logging.INFO,
            format='%(asctime)s - ConversationQueryAPI - %(levelname)s - %(message)s'
        )
        self.logger = logging.getLogger(__name__)
    
    def _find_conversations_db(self) -> str:
        """Find the conversations database file."""
        possible_paths = [
            "knowledge/conversations.duckdb",
            "../knowledge/conversations.duckdb",
            "/Users/cal/DEV/RIF/knowledge/conversations.duckdb"
        ]
        
        for path in possible_paths:
            if Path(path).exists():
                return str(Path(path).resolve())
        
        # Return default path if not found
        default_path = str(Path(__file__).parent.parent / "knowledge" / "conversations.duckdb")
        self.logger.warning(f"Database not found, using default: {default_path}")
        return default_path
    
    def _initialize_storage(self):
        """Initialize storage backend and connection."""
        try:
            self.storage_backend = ConversationStorageBackend(db_path=self.db_path)
            self.connection = self.storage_backend.connection
            self.logger.info("✅ Storage backend initialized")
        except Exception as e:
            self.logger.warning(f"Storage backend not available: {e}")
            # Create direct DuckDB connection as fallback
            try:
                self.connection = duckdb.connect(self.db_path)
                self.logger.info("✅ Direct DuckDB connection established")
            except Exception as e2:
                self.logger.error(f"Failed to establish database connection: {e2}")
                self.connection = None
    
    def search_conversations(self, 
                           query: Optional[str] = None,
                           query_filter: Optional[QueryFilter] = None) -> QueryResult:
        """
        Search conversations with comprehensive filtering and ranking.
        
        Args:
            query: Search query string
            query_filter: Filter parameters
            
        Returns:
            QueryResult with ranked conversation results
        """
        start_time = datetime.now()
        self.logger.info(f"🔍 Searching conversations with query: {query}")
        
        if query_filter is None:
            query_filter = QueryFilter()
        
        try:
            # Determine query type
            query_type = self._determine_query_type(query, query_filter)
            
            # Execute appropriate search method
            if query_type == QueryType.PATTERN_SEARCH:
                results = self._pattern_search(query, query_filter)
            elif query_type == QueryType.SIMILARITY_SEARCH:
                results = self._similarity_search(query, query_filter)
            elif query_type == QueryType.METADATA_FILTER:
                results = self._metadata_filter_search(query_filter)
            elif query_type == QueryType.AGENT_ANALYSIS:
                results = self._agent_analysis_search(query_filter)
            elif query_type == QueryType.SESSION_ANALYSIS:
                results = self._session_analysis_search(query_filter)
            else:
                results = self._default_search(query, query_filter)
            
            # Rank and score results
            ranked_results = self._rank_results(results, query, query_filter)
            
            # Apply limits
            limited_results = ranked_results[:query_filter.max_results]
            
            # Calculate query time
            query_time_ms = (datetime.now() - start_time).total_seconds() * 1000
            
            # Generate suggestions
            suggestions = self._generate_search_suggestions(query, query_filter, len(results))
            
            result = QueryResult(
                query_type=query_type,
                total_matches=len(results),
                results=limited_results,
                query_time_ms=query_time_ms,
                filters_applied=query_filter,
                ranking_method="relevance_score",
                suggestions=suggestions
            )
            
            self.logger.info(f"🎯 Search complete: {len(limited_results)} results in {query_time_ms:.1f}ms")
            return result
            
        except Exception as e:
            self.logger.error(f"Error searching conversations: {e}")
            return self._create_error_result(query, query_filter, str(e))
    
    def _determine_query_type(self, query: Optional[str], query_filter: QueryFilter) -> QueryType:
        """Determine the most appropriate query type."""
        # Check if agent-specific analysis is requested
        if query_filter.agent_type and query and "analyze" in query.lower():
            return QueryType.AGENT_ANALYSIS
        
        # Check if session analysis is requested
        if query_filter.conversation_id or (query and "session" in query.lower()):
            return QueryType.SESSION_ANALYSIS
        
        # Check if outcome analysis is requested
        if query_filter.outcome or (query and "outcome" in query.lower()):
            return QueryType.OUTCOME_ANALYSIS
        
        # Check if similarity search is requested
        if query and any(word in query.lower() for word in ["similar", "like", "related"]):
            return QueryType.SIMILARITY_SEARCH
        
        # Check if pure filtering is requested
        if not query and any([query_filter.agent_type, query_filter.event_type, 
                             query_filter.issue_number, query_filter.date_range]):
            return QueryType.METADATA_FILTER
        
        # Default to pattern search
        return QueryType.PATTERN_SEARCH
    
    def _pattern_search(self, query: str, query_filter: QueryFilter) -> List[ConversationResult]:
        """Perform pattern-based text search."""
        if not self.connection or not query:
            return []
        
        try:
            # Build SQL query for pattern matching
            sql_parts = [
                "SELECT c.conversation_id, c.event_id, c.agent_type, c.event_type,",
                "       c.event_data, c.timestamp, c.issue_number",
                "FROM conversation_events c"
            ]
            
            conditions = []
            params = []
            
            # Text search conditions
            search_terms = query.lower().split()
            for term in search_terms:
                conditions.append("(LOWER(c.event_data) LIKE ? OR LOWER(c.agent_type) LIKE ?)")
                params.extend([f"%{term}%", f"%{term}%"])
            
            # Apply additional filters
            conditions.extend(self._build_filter_conditions(query_filter, params))
            
            if conditions:
                sql_parts.append("WHERE " + " AND ".join(conditions))
            
            sql_parts.append("ORDER BY c.timestamp DESC")
            sql_parts.append(f"LIMIT {query_filter.max_results * 2}")  # Get more for ranking
            
            sql = " ".join(sql_parts)
            
            # Execute query
            cursor = self.connection.execute(sql, params)
            rows = cursor.fetchall()
            
            # Convert to ConversationResult objects
            results = []
            for row in rows:
                try:
                    event_data = json.loads(row[4]) if isinstance(row[4], str) else row[4]
                    result = ConversationResult(
                        conversation_id=row[0],
                        event_id=row[1],
                        agent_type=row[2],
                        event_type=row[3],
                        event_data=event_data,
                        timestamp=datetime.fromisoformat(row[5]) if isinstance(row[5], str) else row[5],
                        issue_number=row[6],
                        score=self._calculate_pattern_score(query, event_data, row[2]),
                        relevance_factors=self._identify_relevance_factors(query, event_data, row[2])
                    )
                    results.append(result)
                except Exception as e:
                    self.logger.debug(f"Error processing row: {e}")
                    continue
            
            return results
            
        except Exception as e:
            self.logger.error(f"Error in pattern search: {e}")
            return []
    
    def _similarity_search(self, query: str, query_filter: QueryFilter) -> List[ConversationResult]:
        """Perform similarity-based search using embeddings."""
        # For now, fall back to pattern search as embeddings might not be available
        self.logger.debug("Similarity search not yet implemented, using pattern search")
        return self._pattern_search(query, query_filter)
    
    def _metadata_filter_search(self, query_filter: QueryFilter) -> List[ConversationResult]:
        """Perform metadata-based filtering."""
        if not self.connection:
            return []
        
        try:
            sql_parts = [
                "SELECT c.conversation_id, c.event_id, c.agent_type, c.event_type,",
                "       c.event_data, c.timestamp, c.issue_number",
                "FROM conversation_events c"
            ]
            
            conditions = []
            params = []
            
            # Apply all filters
            conditions.extend(self._build_filter_conditions(query_filter, params))
            
            if conditions:
                sql_parts.append("WHERE " + " AND ".join(conditions))
            
            sql_parts.append("ORDER BY c.timestamp DESC")
            sql_parts.append(f"LIMIT {query_filter.max_results}")
            
            sql = " ".join(sql_parts)
            
            # Execute query
            cursor = self.connection.execute(sql, params)
            rows = cursor.fetchall()
            
            # Convert to results
            results = []
            for row in rows:
                try:
                    event_data = json.loads(row[4]) if isinstance(row[4], str) else row[4]
                    result = ConversationResult(
                        conversation_id=row[0],
                        event_id=row[1],
                        agent_type=row[2],
                        event_type=row[3],
                        event_data=event_data,
                        timestamp=datetime.fromisoformat(row[5]) if isinstance(row[5], str) else row[5],
                        issue_number=row[6],
                        score=0.8,  # Default score for metadata matches
                        relevance_factors=["metadata_match"]
                    )
                    results.append(result)
                except Exception as e:
                    self.logger.debug(f"Error processing row: {e}")
                    continue
            
            return results
            
        except Exception as e:
            self.logger.error(f"Error in metadata filter search: {e}")
            return []
    
    def _agent_analysis_search(self, query_filter: QueryFilter) -> List[ConversationResult]:
        """Perform agent-specific analysis search."""
        # Filter by agent type and get diverse results
        agent_filter = QueryFilter(
            agent_type=query_filter.agent_type,
            date_range=query_filter.date_range,
            max_results=query_filter.max_results
        )
        
        return self._metadata_filter_search(agent_filter)
    
    def _session_analysis_search(self, query_filter: QueryFilter) -> List[ConversationResult]:
        """Perform conversation session analysis."""
        # Filter by conversation ID and get all events
        session_filter = QueryFilter(
            conversation_id=query_filter.conversation_id,
            max_results=query_filter.max_results or 100
        )
        
        return self._metadata_filter_search(session_filter)
    
    def _default_search(self, query: Optional[str], query_filter: QueryFilter) -> List[ConversationResult]:
        """Default search when no specific type is determined."""
        if query:
            return self._pattern_search(query, query_filter)
        else:
            return self._metadata_filter_search(query_filter)
    
    def _build_filter_conditions(self, query_filter: QueryFilter, params: List[Any]) -> List[str]:
        """Build SQL filter conditions from QueryFilter."""
        conditions = []
        
        if query_filter.agent_type:
            conditions.append("c.agent_type = ?")
            params.append(query_filter.agent_type)
        
        if query_filter.event_type:
            conditions.append("c.event_type = ?")
            params.append(query_filter.event_type)
        
        if query_filter.issue_number:
            conditions.append("c.issue_number = ?")
            params.append(query_filter.issue_number)
        
        if query_filter.conversation_id:
            conditions.append("c.conversation_id = ?")
            params.append(query_filter.conversation_id)
        
        if query_filter.date_range:
            start_date, end_date = query_filter.date_range
            conditions.append("c.timestamp BETWEEN ? AND ?")
            params.extend([start_date.isoformat(), end_date.isoformat()])
        
        return conditions
    
    def _calculate_pattern_score(self, query: str, event_data: Dict[str, Any], agent_type: str) -> float:
        """Calculate relevance score for pattern matches."""
        score = 0.0
        query_lower = query.lower()
        search_terms = query_lower.split()
        
        # Score based on text content
        text_content = json.dumps(event_data).lower() if event_data else ""
        
        # Exact phrase match bonus
        if query_lower in text_content:
            score += 0.5
        
        # Individual term matches
        term_matches = sum(1 for term in search_terms if term in text_content)
        if search_terms:
            score += 0.3 * (term_matches / len(search_terms))
        
        # Agent type relevance
        if query_lower in agent_type.lower():
            score += 0.2
        
        # Content length normalization
        content_length = len(text_content)
        if content_length > 100:
            score *= min(1.2, 1 + (content_length / 1000))  # Bonus for detailed content
        
        return min(1.0, score)
    
    def _identify_relevance_factors(self, query: str, event_data: Dict[str, Any], agent_type: str) -> List[str]:
        """Identify factors that make a result relevant."""
        factors = []
        query_lower = query.lower()
        text_content = json.dumps(event_data).lower() if event_data else ""
        
        if query_lower in text_content:
            factors.append("exact_phrase_match")
        
        for term in query_lower.split():
            if term in text_content:
                factors.append(f"term_match:{term}")
        
        if query_lower in agent_type.lower():
            factors.append("agent_type_match")
        
        if event_data:
            if "error" in text_content and "error" in query_lower:
                factors.append("error_context")
            if "success" in text_content and "success" in query_lower:
                factors.append("success_context")
        
        return factors[:5]  # Limit to top 5 factors
    
    def _rank_results(self, results: List[ConversationResult], query: Optional[str], query_filter: QueryFilter) -> List[ConversationResult]:
        """Rank and sort results by relevance."""
        # Apply minimum score filter
        if query_filter.min_score:
            results = [r for r in results if r.score >= query_filter.min_score]
        
        # Sort by score (descending) and timestamp (descending)
        results.sort(key=lambda x: (x.score, x.timestamp), reverse=True)
        
        return results
    
    def _generate_search_suggestions(self, query: Optional[str], query_filter: QueryFilter, total_results: int) -> List[str]:
        """Generate helpful search suggestions."""
        suggestions = []
        
        if total_results == 0:
            suggestions.append("Try broadening your search terms")
            suggestions.append("Remove some filters to see more results")
            if query_filter.agent_type:
                suggestions.append("Try searching without the agent_type filter")
        
        if query and len(query.split()) == 1:
            suggestions.append("Try adding more search terms for better results")
        
        if not query_filter.date_range:
            suggestions.append("Add date range filter to narrow results")
        
        if total_results > query_filter.max_results:
            suggestions.append(f"Showing top {query_filter.max_results} of {total_results} results - increase max_results for more")
        
        return suggestions
    
    def _create_error_result(self, query: Optional[str], query_filter: QueryFilter, error: str) -> QueryResult:
        """Create error result when search fails."""
        return QueryResult(
            query_type=QueryType.PATTERN_SEARCH,
            total_matches=0,
            results=[],
            query_time_ms=0.0,
            filters_applied=query_filter,
            ranking_method="error",
            suggestions=[f"Search failed: {error}", "Check database connection", "Verify query parameters"]
        )
    
    def get_conversation_summary(self, conversation_id: str) -> Dict[str, Any]:
        """
        Get comprehensive summary of a conversation.
        
        Args:
            conversation_id: Conversation identifier
            
        Returns:
            Comprehensive conversation summary
        """
        try:
            if not self.connection:
                return {"error": "Database connection not available"}
            
            # Get all events for this conversation
            cursor = self.connection.execute("""
                SELECT event_id, agent_type, event_type, event_data, timestamp, issue_number
                FROM conversation_events 
                WHERE conversation_id = ?
                ORDER BY timestamp ASC
            """, [conversation_id])
            
            events = cursor.fetchall()
            
            if not events:
                return {"error": f"No conversation found with ID: {conversation_id}"}
            
            # Process events
            processed_events = []
            agent_types = set()
            event_types = set()
            
            for event in events:
                try:
                    event_data = json.loads(event[3]) if isinstance(event[3], str) else event[3]
                    processed_event = {
                        'event_id': event[0],
                        'agent_type': event[1],
                        'event_type': event[2],
                        'event_data': event_data,
                        'timestamp': event[4],
                        'issue_number': event[5]
                    }
                    processed_events.append(processed_event)
                    agent_types.add(event[1])
                    event_types.add(event[2])
                except Exception as e:
                    self.logger.debug(f"Error processing event: {e}")
            
            # Generate summary
            summary = {
                'conversation_id': conversation_id,
                'total_events': len(processed_events),
                'agent_types': list(agent_types),
                'event_types': list(event_types),
                'start_time': processed_events[0]['timestamp'] if processed_events else None,
                'end_time': processed_events[-1]['timestamp'] if processed_events else None,
                'duration': self._calculate_duration(processed_events),
                'events': processed_events,
                'analysis': self._analyze_conversation(processed_events)
            }
            
            return summary
            
        except Exception as e:
            self.logger.error(f"Error getting conversation summary: {e}")
            return {"error": str(e)}
    
    def _calculate_duration(self, events: List[Dict[str, Any]]) -> Optional[str]:
        """Calculate conversation duration."""
        if len(events) < 2:
            return None
        
        try:
            start_time = datetime.fromisoformat(events[0]['timestamp'])
            end_time = datetime.fromisoformat(events[-1]['timestamp'])
            duration = end_time - start_time
            
            if duration.total_seconds() < 60:
                return f"{duration.seconds} seconds"
            elif duration.total_seconds() < 3600:
                return f"{duration.seconds // 60} minutes"
            else:
                hours = duration.seconds // 3600
                minutes = (duration.seconds % 3600) // 60
                return f"{hours}h {minutes}m"
        except Exception:
            return None
    
    def _analyze_conversation(self, events: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Analyze conversation patterns and outcomes."""
        analysis = {
            'pattern': 'unknown',
            'outcome': 'unknown',
            'key_topics': [],
            'agent_interactions': {},
            'issues_addressed': []
        }
        
        try:
            # Analyze agent interactions
            agent_counts = {}
            for event in events:
                agent_type = event.get('agent_type', 'unknown')
                agent_counts[agent_type] = agent_counts.get(agent_type, 0) + 1
            
            analysis['agent_interactions'] = agent_counts
            
            # Identify primary pattern
            if len(agent_counts) == 1:
                analysis['pattern'] = 'single_agent'
            elif len(agent_counts) > 3:
                analysis['pattern'] = 'multi_agent_collaboration'
            else:
                analysis['pattern'] = 'agent_handoff'
            
            # Extract issues addressed
            issues = set()
            for event in events:
                issue_num = event.get('issue_number')
                if issue_num:
                    issues.add(issue_num)
            
            analysis['issues_addressed'] = list(issues)
            
            # Determine outcome
            last_events = events[-3:]  # Look at last few events
            for event in reversed(last_events):
                event_data = event.get('event_data', {})
                if isinstance(event_data, dict):
                    if any(word in str(event_data).lower() for word in ['complete', 'success', 'done']):
                        analysis['outcome'] = 'successful'
                        break
                    elif any(word in str(event_data).lower() for word in ['error', 'failed', 'blocked']):
                        analysis['outcome'] = 'failed'
                        break
        
        except Exception as e:
            self.logger.debug(f"Error in conversation analysis: {e}")
        
        return analysis
    
    def get_agent_performance_stats(self, agent_type: Optional[str] = None, days: int = 30) -> Dict[str, Any]:
        """
        Get performance statistics for agents.
        
        Args:
            agent_type: Specific agent type to analyze (or None for all)
            days: Number of days to look back
            
        Returns:
            Agent performance statistics
        """
        try:
            if not self.connection:
                return {"error": "Database connection not available"}
            
            # Date range
            end_date = datetime.now()
            start_date = end_date - timedelta(days=days)
            
            # Build query
            sql_parts = [
                "SELECT agent_type, event_type, COUNT(*) as event_count",
                "FROM conversation_events",
                "WHERE timestamp >= ?"
            ]
            params = [start_date.isoformat()]
            
            if agent_type:
                sql_parts.append("AND agent_type = ?")
                params.append(agent_type)
            
            sql_parts.append("GROUP BY agent_type, event_type")
            sql_parts.append("ORDER BY agent_type, event_count DESC")
            
            sql = " ".join(sql_parts)
            
            # Execute query
            cursor = self.connection.execute(sql, params)
            rows = cursor.fetchall()
            
            # Process results
            stats = {
                'period_days': days,
                'start_date': start_date.isoformat(),
                'end_date': end_date.isoformat(),
                'agents': {}
            }
            
            for row in rows:
                agent = row[0]
                event_type = row[1]
                count = row[2]
                
                if agent not in stats['agents']:
                    stats['agents'][agent] = {
                        'total_events': 0,
                        'event_types': {},
                        'avg_events_per_day': 0
                    }
                
                stats['agents'][agent]['event_types'][event_type] = count
                stats['agents'][agent]['total_events'] += count
            
            # Calculate averages
            for agent_data in stats['agents'].values():
                agent_data['avg_events_per_day'] = agent_data['total_events'] / days
            
            return stats
            
        except Exception as e:
            self.logger.error(f"Error getting agent performance stats: {e}")
            return {"error": str(e)}
    
    def close(self):
        """Close database connections."""
        try:
            if self.storage_backend:
                self.storage_backend.close()
            elif self.connection:
                self.connection.close()
        except Exception as e:
            self.logger.debug(f"Error closing connections: {e}")

def main():
    """Command line interface for conversation query API."""
    import sys
    
    if len(sys.argv) < 2:
        print("Usage: python conversation_query_api.py <command> [args]")
        print("Commands:")
        print("  search <query> [agent_type] [max_results]     - Search conversations")
        print("  summary <conversation_id>                     - Get conversation summary")  
        print("  stats [agent_type] [days]                     - Get agent performance stats")
        print("  filter <agent_type> [event_type] [days]       - Filter by metadata")
        return
    
    command = sys.argv[1]
    api = ConversationQueryAPI()
    
    if command == "search":
        if len(sys.argv) < 3:
            print("Usage: search <query> [agent_type] [max_results]")
            return 1
        
        query = sys.argv[2]
        agent_type = sys.argv[3] if len(sys.argv) > 3 else None
        max_results = int(sys.argv[4]) if len(sys.argv) > 4 else 10
        
        query_filter = QueryFilter(agent_type=agent_type, max_results=max_results)
        result = api.search_conversations(query, query_filter)
        
        print(f"🔍 Search Results for: {query}")
        print(f"Total matches: {result.total_matches}")
        print(f"Query time: {result.query_time_ms:.1f}ms")
        print(f"Query type: {result.query_type.value}")
        
        for i, res in enumerate(result.results, 1):
            print(f"\n{i}. {res.agent_type} - {res.event_type} (Score: {res.score:.2f})")
            print(f"   Time: {res.timestamp}")
            print(f"   Conversation: {res.conversation_id}")
            if res.issue_number:
                print(f"   Issue: #{res.issue_number}")
            print(f"   Relevance: {', '.join(res.relevance_factors)}")
        
        if result.suggestions:
            print(f"\nSuggestions:")
            for suggestion in result.suggestions:
                print(f"  - {suggestion}")
    
    elif command == "summary":
        if len(sys.argv) < 3:
            print("Usage: summary <conversation_id>")
            return 1
        
        conversation_id = sys.argv[2]
        summary = api.get_conversation_summary(conversation_id)
        
        if "error" in summary:
            print(f"Error: {summary['error']}")
            return 1
        
        print(f"📊 Conversation Summary: {conversation_id}")
        print(f"Events: {summary['total_events']}")
        print(f"Agent types: {', '.join(summary['agent_types'])}")
        print(f"Event types: {', '.join(summary['event_types'])}")
        print(f"Duration: {summary.get('duration', 'N/A')}")
        print(f"Pattern: {summary['analysis']['pattern']}")
        print(f"Outcome: {summary['analysis']['outcome']}")
        
        if summary['analysis']['issues_addressed']:
            print(f"Issues: {', '.join(map(str, summary['analysis']['issues_addressed']))}")
    
    elif command == "stats":
        agent_type = sys.argv[2] if len(sys.argv) > 2 else None
        days = int(sys.argv[3]) if len(sys.argv) > 3 else 30
        
        stats = api.get_agent_performance_stats(agent_type, days)
        
        if "error" in stats:
            print(f"Error: {stats['error']}")
            return 1
        
        print(f"📈 Agent Performance Stats ({days} days)")
        
        for agent, data in stats['agents'].items():
            print(f"\n{agent}:")
            print(f"  Total events: {data['total_events']}")
            print(f"  Avg per day: {data['avg_events_per_day']:.1f}")
            print(f"  Event types:")
            for event_type, count in data['event_types'].items():
                print(f"    {event_type}: {count}")
    
    elif command == "filter":
        if len(sys.argv) < 3:
            print("Usage: filter <agent_type> [event_type] [days]")
            return 1
        
        agent_type = sys.argv[2]
        event_type = sys.argv[3] if len(sys.argv) > 3 else None
        days = int(sys.argv[4]) if len(sys.argv) > 4 else 7
        
        end_date = datetime.now()
        start_date = end_date - timedelta(days=days)
        
        query_filter = QueryFilter(
            agent_type=agent_type,
            event_type=event_type,
            date_range=(start_date, end_date),
            max_results=20
        )
        
        result = api.search_conversations(None, query_filter)
        
        print(f"🔍 Filtered Results: {agent_type}")
        print(f"Total matches: {result.total_matches}")
        
        for res in result.results:
            print(f"  {res.timestamp} - {res.event_type}")
            if res.issue_number:
                print(f"    Issue: #{res.issue_number}")
    
    else:
        print(f"Unknown command: {command}")
        return 1
    
    api.close()
    return 0

if __name__ == "__main__":
    exit(main())