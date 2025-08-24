"""
Pattern detection system for agent conversations.

Analyzes agent conversations to identify recurring patterns, successful
strategies, common failure modes, and learning opportunities.
"""

import json
import statistics
from datetime import datetime, timedelta
from typing import Dict, List, Any, Optional, Tuple
from collections import defaultdict, Counter
import logging

from .storage_backend import ConversationStorageBackend

logger = logging.getLogger(__name__)


class ConversationPatternDetector:
    """
    Pattern detection and analysis system for agent conversations.
    
    Identifies recurring patterns, successful strategies, failure modes,
    and learning opportunities from agent interaction data.
    """
    
    def __init__(self, storage_backend: Optional[ConversationStorageBackend] = None):
        """
        Initialize pattern detector.
        
        Args:
            storage_backend: Storage backend (creates default if None)
        """
        self.storage = storage_backend or ConversationStorageBackend()
        
        # Pattern detection configuration
        self.config = {
            'min_pattern_occurrences': 3,
            'success_threshold': 0.7,
            'failure_threshold': 0.3,
            'confidence_threshold': 0.6,
            'analysis_window_days': 30
        }
        
        # Analysis cache
        self.pattern_cache = {}
        self.last_analysis_time = None
    
    def detect_success_patterns(self,
                              agent_type: Optional[str] = None,
                              days_back: int = 30,
                              min_occurrences: int = 3) -> List[Dict[str, Any]]:
        """
        Detect patterns that lead to successful outcomes.
        
        Args:
            agent_type: Filter by agent type
            days_back: Days to look back for analysis
            min_occurrences: Minimum occurrences to consider a pattern
            
        Returns:
            List of successful patterns with success rates
        """
        try:
            start_date = datetime.now() - timedelta(days=days_back)
            
            # Get successful conversations
            sql = """
                SELECT 
                    cm.agent_type,
                    cm.conversation_id,
                    cm.total_events,
                    cm.decision_count,
                    cm.error_count,
                    ce.event_type,
                    ce.event_data
                FROM conversation_metadata cm
                JOIN conversation_events ce ON cm.conversation_id = ce.conversation_id
                WHERE cm.success = true 
                  AND cm.start_timestamp >= ?
            """
            
            params = [start_date]
            
            if agent_type:
                sql += " AND cm.agent_type = ?"
                params.append(agent_type)
            
            sql += " ORDER BY cm.start_timestamp DESC"
            
            result = self.storage.connection.execute(sql, params)
            
            # Group by pattern characteristics
            pattern_groups = defaultdict(list)
            
            for row in result.fetchall():
                row_dict = dict(zip([col[0] for col in result.description], row))
                event_data = json.loads(row_dict['event_data']) if row_dict['event_data'] else {}
                
                # Extract pattern features
                pattern_key = self._extract_success_pattern_key(
                    row_dict['agent_type'],
                    row_dict['event_type'], 
                    event_data
                )
                
                pattern_groups[pattern_key].append(row_dict)
            
            # Analyze patterns with sufficient occurrences
            success_patterns = []
            
            for pattern_key, occurrences in pattern_groups.items():
                if len(occurrences) >= min_occurrences:
                    pattern_analysis = self._analyze_success_pattern(pattern_key, occurrences)
                    if pattern_analysis:
                        success_patterns.append(pattern_analysis)
            
            # Sort by success rate and occurrence count
            success_patterns.sort(key=lambda p: (p['success_rate'], p['occurrence_count']), reverse=True)
            
            return success_patterns
            
        except Exception as e:
            logger.error(f"Success pattern detection failed: {e}")
            return []
    
    def detect_failure_patterns(self,
                              agent_type: Optional[str] = None,
                              days_back: int = 30,
                              min_occurrences: int = 2) -> List[Dict[str, Any]]:
        """
        Detect patterns that lead to failures or poor outcomes.
        
        Args:
            agent_type: Filter by agent type
            days_back: Days to look back for analysis
            min_occurrences: Minimum occurrences to consider a pattern
            
        Returns:
            List of failure patterns with failure rates
        """
        try:
            start_date = datetime.now() - timedelta(days=days_back)
            
            # Get failed conversations and errors
            sql = """
                SELECT DISTINCT
                    cm.agent_type,
                    cm.conversation_id,
                    cm.success,
                    ce.error_type,
                    ce.error_message,
                    ce.pattern_signature,
                    ce.resolution_success,
                    ev.event_type,
                    ev.event_data
                FROM conversation_metadata cm
                JOIN conversation_errors ce ON cm.conversation_id = ce.conversation_id
                LEFT JOIN conversation_events ev ON cm.conversation_id = ev.conversation_id
                WHERE (cm.success = false OR ce.resolution_success = false)
                  AND cm.start_timestamp >= ?
            """
            
            params = [start_date]
            
            if agent_type:
                sql += " AND cm.agent_type = ?"
                params.append(agent_type)
            
            result = self.storage.connection.execute(sql, params)
            
            # Group by failure pattern characteristics
            pattern_groups = defaultdict(list)
            
            for row in result.fetchall():
                row_dict = dict(zip([col[0] for col in result.description], row))
                
                # Use error pattern signature as primary grouping
                pattern_key = row_dict['pattern_signature']
                if pattern_key:
                    pattern_groups[pattern_key].append(row_dict)
                
                # Also group by event patterns leading to errors
                if row_dict['event_data']:
                    event_data = json.loads(row_dict['event_data'])
                    event_pattern_key = self._extract_failure_pattern_key(
                        row_dict['agent_type'],
                        row_dict['event_type'],
                        event_data
                    )
                    if event_pattern_key:
                        pattern_groups[f"event_{event_pattern_key}"].append(row_dict)
            
            # Analyze failure patterns
            failure_patterns = []
            
            for pattern_key, occurrences in pattern_groups.items():
                if len(occurrences) >= min_occurrences:
                    pattern_analysis = self._analyze_failure_pattern(pattern_key, occurrences)
                    if pattern_analysis:
                        failure_patterns.append(pattern_analysis)
            
            # Sort by failure rate and occurrence count
            failure_patterns.sort(key=lambda p: (p['failure_rate'], p['occurrence_count']), reverse=True)
            
            return failure_patterns
            
        except Exception as e:
            logger.error(f"Failure pattern detection failed: {e}")
            return []
    
    def analyze_decision_patterns(self,
                                agent_type: Optional[str] = None,
                                days_back: int = 30) -> Dict[str, Any]:
        """
        Analyze decision-making patterns and their outcomes.
        
        Args:
            agent_type: Filter by agent type
            days_back: Days to look back for analysis
            
        Returns:
            Analysis of decision patterns and success rates
        """
        try:
            start_date = datetime.now() - timedelta(days=days_back)
            
            sql = """
                SELECT 
                    agent_type,
                    decision_point,
                    chosen_option,
                    confidence_score,
                    outcome,
                    learning_value,
                    rationale
                FROM agent_decisions
                WHERE created_at >= ?
            """
            
            params = [start_date]
            
            if agent_type:
                sql += " AND agent_type = ?"
                params.append(agent_type)
            
            result = self.storage.connection.execute(sql, params)
            decisions = [dict(zip([col[0] for col in result.description], row)) for row in result.fetchall()]
            
            if not decisions:
                return {}
            
            # Analyze decision patterns
            analysis = {
                'total_decisions': len(decisions),
                'agent_breakdown': self._analyze_decisions_by_agent(decisions),
                'confidence_analysis': self._analyze_decision_confidence(decisions),
                'outcome_analysis': self._analyze_decision_outcomes(decisions),
                'pattern_insights': self._extract_decision_insights(decisions)
            }
            
            return analysis
            
        except Exception as e:
            logger.error(f"Decision pattern analysis failed: {e}")
            return {}
    
    def _extract_success_pattern_key(self,
                                   agent_type: str,
                                   event_type: str,
                                   event_data: Dict[str, Any]) -> str:
        """Extract pattern key for successful behaviors."""
        key_parts = [agent_type, event_type]
        
        # Add significant event data features
        if 'tool_name' in event_data:
            key_parts.append(f"tool_{event_data['tool_name']}")
        
        if 'status' in event_data:
            key_parts.append(f"status_{event_data['status']}")
        
        if event_type == 'decision' and 'decision_type' in event_data:
            key_parts.append(f"decision_{event_data['decision_type']}")
        
        return "_".join(key_parts)
    
    def _extract_failure_pattern_key(self,
                                   agent_type: str,
                                   event_type: str,
                                   event_data: Dict[str, Any]) -> Optional[str]:
        """Extract pattern key for failure behaviors."""
        if event_type != 'tool_error' and event_type != 'error':
            return None
        
        key_parts = [agent_type, event_type]
        
        if 'tool_name' in event_data:
            key_parts.append(f"tool_{event_data['tool_name']}")
        
        if 'error_type' in event_data:
            key_parts.append(f"error_{event_data['error_type']}")
        
        return "_".join(key_parts)
    
    def _analyze_success_pattern(self,
                               pattern_key: str,
                               occurrences: List[Dict[str, Any]]) -> Optional[Dict[str, Any]]:
        """Analyze a success pattern."""
        try:
            total_conversations = len(set(occ['conversation_id'] for occ in occurrences))
            successful_conversations = len(set(
                occ['conversation_id'] for occ in occurrences 
                if self._is_conversation_successful(occ['conversation_id'])
            ))
            
            success_rate = successful_conversations / max(total_conversations, 1)
            
            if success_rate < self.config['success_threshold']:
                return None
            
            # Calculate additional metrics
            avg_events = statistics.mean([occ['total_events'] for occ in occurrences if occ['total_events']])
            avg_decisions = statistics.mean([occ['decision_count'] for occ in occurrences if occ['decision_count']])
            avg_errors = statistics.mean([occ['error_count'] for occ in occurrences if occ['error_count']])
            
            return {
                'pattern_key': pattern_key,
                'pattern_type': 'success',
                'occurrence_count': len(occurrences),
                'conversation_count': total_conversations,
                'success_rate': success_rate,
                'avg_events_per_conversation': avg_events,
                'avg_decisions_per_conversation': avg_decisions,
                'avg_errors_per_conversation': avg_errors,
                'confidence': min(success_rate * (total_conversations / 10), 1.0),  # Confidence based on sample size
                'recommendation': f"Continue using pattern '{pattern_key}' - shows {success_rate:.1%} success rate"
            }
            
        except Exception as e:
            logger.warning(f"Error analyzing success pattern {pattern_key}: {e}")
            return None
    
    def _analyze_failure_pattern(self,
                               pattern_key: str,
                               occurrences: List[Dict[str, Any]]) -> Optional[Dict[str, Any]]:
        """Analyze a failure pattern."""
        try:
            total_conversations = len(set(occ['conversation_id'] for occ in occurrences))
            failed_conversations = len(set(
                occ['conversation_id'] for occ in occurrences 
                if not self._is_conversation_successful(occ['conversation_id'])
            ))
            
            failure_rate = failed_conversations / max(total_conversations, 1)
            
            if failure_rate < self.config['failure_threshold']:
                return None
            
            # Analyze resolution attempts
            resolution_attempts = sum(1 for occ in occurrences if occ.get('resolution_attempted'))
            successful_resolutions = sum(1 for occ in occurrences if occ.get('resolution_success'))
            resolution_success_rate = successful_resolutions / max(resolution_attempts, 1)
            
            # Get most common error types
            error_types = [occ['error_type'] for occ in occurrences if occ.get('error_type')]
            most_common_errors = Counter(error_types).most_common(3)
            
            return {
                'pattern_key': pattern_key,
                'pattern_type': 'failure',
                'occurrence_count': len(occurrences),
                'conversation_count': total_conversations,
                'failure_rate': failure_rate,
                'resolution_attempts': resolution_attempts,
                'resolution_success_rate': resolution_success_rate,
                'common_error_types': most_common_errors,
                'risk_level': 'high' if failure_rate > 0.7 else 'medium' if failure_rate > 0.4 else 'low',
                'recommendation': f"Investigate and mitigate pattern '{pattern_key}' - shows {failure_rate:.1%} failure rate"
            }
            
        except Exception as e:
            logger.warning(f"Error analyzing failure pattern {pattern_key}: {e}")
            return None
    
    def _is_conversation_successful(self, conversation_id: str) -> bool:
        """Check if a conversation was successful."""
        try:
            result = self.storage.connection.execute("""
                SELECT success FROM conversation_metadata WHERE conversation_id = ?
            """, [conversation_id])
            
            row = result.fetchone()
            return row[0] if row else False
            
        except Exception:
            return False
    
    def _analyze_decisions_by_agent(self, decisions: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Analyze decision patterns by agent type."""
        agent_stats = defaultdict(lambda: {
            'total': 0,
            'outcomes': Counter(),
            'avg_confidence': 0.0,
            'confidence_scores': []
        })
        
        for decision in decisions:
            agent_type = decision['agent_type']
            agent_stats[agent_type]['total'] += 1
            
            if decision['outcome']:
                agent_stats[agent_type]['outcomes'][decision['outcome']] += 1
            
            if decision['confidence_score']:
                agent_stats[agent_type]['confidence_scores'].append(decision['confidence_score'])
        
        # Calculate averages
        for agent_type, stats in agent_stats.items():
            if stats['confidence_scores']:
                stats['avg_confidence'] = statistics.mean(stats['confidence_scores'])
                stats['confidence_std'] = statistics.stdev(stats['confidence_scores']) if len(stats['confidence_scores']) > 1 else 0
                del stats['confidence_scores']  # Clean up raw data
        
        return dict(agent_stats)
    
    def _analyze_decision_confidence(self, decisions: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Analyze decision confidence patterns."""
        confidence_scores = [d['confidence_score'] for d in decisions if d['confidence_score'] is not None]
        
        if not confidence_scores:
            return {}
        
        return {
            'avg_confidence': statistics.mean(confidence_scores),
            'median_confidence': statistics.median(confidence_scores),
            'confidence_std': statistics.stdev(confidence_scores) if len(confidence_scores) > 1 else 0,
            'high_confidence_decisions': sum(1 for score in confidence_scores if score >= 0.8),
            'low_confidence_decisions': sum(1 for score in confidence_scores if score <= 0.3),
            'confidence_distribution': {
                'very_low': sum(1 for score in confidence_scores if score <= 0.2),
                'low': sum(1 for score in confidence_scores if 0.2 < score <= 0.4),
                'medium': sum(1 for score in confidence_scores if 0.4 < score <= 0.6),
                'high': sum(1 for score in confidence_scores if 0.6 < score <= 0.8),
                'very_high': sum(1 for score in confidence_scores if score > 0.8)
            }
        }
    
    def _analyze_decision_outcomes(self, decisions: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Analyze decision outcome patterns."""
        outcomes = [d['outcome'] for d in decisions if d['outcome']]
        outcome_counts = Counter(outcomes)
        
        # Calculate success rate
        successful_outcomes = ['success', 'partial']
        failed_outcomes = ['failure', 'error']
        
        success_count = sum(outcome_counts.get(outcome, 0) for outcome in successful_outcomes)
        failure_count = sum(outcome_counts.get(outcome, 0) for outcome in failed_outcomes)
        total_with_outcomes = success_count + failure_count
        
        return {
            'outcome_distribution': dict(outcome_counts),
            'success_rate': success_count / max(total_with_outcomes, 1),
            'failure_rate': failure_count / max(total_with_outcomes, 1),
            'decisions_with_outcomes': total_with_outcomes,
            'decisions_without_outcomes': len(decisions) - total_with_outcomes
        }
    
    def _extract_decision_insights(self, decisions: List[Dict[str, Any]]) -> List[str]:
        """Extract actionable insights from decision patterns."""
        insights = []
        
        # Confidence insights
        confidence_scores = [d['confidence_score'] for d in decisions if d['confidence_score'] is not None]
        if confidence_scores:
            avg_confidence = statistics.mean(confidence_scores)
            if avg_confidence < 0.5:
                insights.append("Low average decision confidence suggests need for better decision support")
            elif avg_confidence > 0.8:
                insights.append("High decision confidence indicates good decision-making processes")
        
        # Outcome insights
        outcomes = [d['outcome'] for d in decisions if d['outcome']]
        if outcomes:
            outcome_counts = Counter(outcomes)
            total_outcomes = len(outcomes)
            
            success_rate = sum(outcome_counts.get(outcome, 0) for outcome in ['success', 'partial']) / total_outcomes
            if success_rate < 0.6:
                insights.append("Low success rate suggests need for decision process improvements")
            elif success_rate > 0.85:
                insights.append("High success rate indicates effective decision-making")
        
        # Agent-specific insights
        agent_performance = {}
        for decision in decisions:
            agent_type = decision['agent_type']
            if agent_type not in agent_performance:
                agent_performance[agent_type] = []
            if decision['outcome'] in ['success', 'partial']:
                agent_performance[agent_type].append(1)
            elif decision['outcome'] in ['failure', 'error']:
                agent_performance[agent_type].append(0)
        
        for agent_type, outcomes in agent_performance.items():
            if len(outcomes) >= 5:  # Enough data for insight
                success_rate = sum(outcomes) / len(outcomes)
                if success_rate < 0.5:
                    insights.append(f"{agent_type} shows low decision success rate ({success_rate:.1%})")
                elif success_rate > 0.9:
                    insights.append(f"{agent_type} shows excellent decision success rate ({success_rate:.1%})")
        
        return insights
    
    def generate_learning_recommendations(self,
                                        agent_type: Optional[str] = None,
                                        days_back: int = 30) -> List[Dict[str, Any]]:
        """
        Generate learning recommendations based on pattern analysis.
        
        Args:
            agent_type: Filter by agent type
            days_back: Days to analyze
            
        Returns:
            List of learning recommendations
        """
        try:
            recommendations = []
            
            # Get success and failure patterns
            success_patterns = self.detect_success_patterns(agent_type, days_back)
            failure_patterns = self.detect_failure_patterns(agent_type, days_back)
            decision_analysis = self.analyze_decision_patterns(agent_type, days_back)
            
            # Success pattern recommendations
            for pattern in success_patterns[:3]:  # Top 3 success patterns
                recommendations.append({
                    'type': 'replicate_success',
                    'priority': 'high',
                    'title': f"Replicate successful pattern: {pattern['pattern_key']}",
                    'description': f"This pattern shows {pattern['success_rate']:.1%} success rate with {pattern['conversation_count']} conversations",
                    'action': f"Encourage continued use of {pattern['pattern_key']} approach",
                    'confidence': pattern['confidence']
                })
            
            # Failure pattern recommendations
            for pattern in failure_patterns[:2]:  # Top 2 failure patterns
                recommendations.append({
                    'type': 'mitigate_failure',
                    'priority': 'high' if pattern['failure_rate'] > 0.7 else 'medium',
                    'title': f"Address failure pattern: {pattern['pattern_key']}",
                    'description': f"This pattern shows {pattern['failure_rate']:.1%} failure rate with {pattern['conversation_count']} conversations",
                    'action': f"Investigate root cause and develop mitigation for {pattern['pattern_key']}",
                    'risk_level': pattern['risk_level']
                })
            
            # Decision-making recommendations
            if decision_analysis and 'confidence_analysis' in decision_analysis:
                conf_analysis = decision_analysis['confidence_analysis']
                if conf_analysis.get('avg_confidence', 0) < 0.6:
                    recommendations.append({
                        'type': 'improve_decisions',
                        'priority': 'medium',
                        'title': 'Improve decision confidence',
                        'description': f"Average decision confidence is {conf_analysis['avg_confidence']:.2f}, below recommended 0.6",
                        'action': 'Provide better decision support tools and processes'
                    })
            
            # Sort by priority and confidence
            priority_order = {'high': 3, 'medium': 2, 'low': 1}
            recommendations.sort(key=lambda r: (
                priority_order.get(r.get('priority', 'low'), 0),
                r.get('confidence', 0.5)
            ), reverse=True)
            
            return recommendations
            
        except Exception as e:
            logger.error(f"Learning recommendation generation failed: {e}")
            return []
    
    def get_pattern_detection_summary(self) -> Dict[str, Any]:
        """Get summary of pattern detection capabilities and recent analysis."""
        try:
            # Get recent analysis counts
            recent_date = datetime.now() - timedelta(days=7)
            
            stats = {
                'config': self.config,
                'cache_size': len(self.pattern_cache),
                'last_analysis': self.last_analysis_time.isoformat() if self.last_analysis_time else None,
                'recent_analysis': {}
            }
            
            # Count recent conversations by type
            result = self.storage.connection.execute("""
                SELECT 
                    agent_type,
                    COUNT(*) as conversation_count,
                    SUM(CASE WHEN success = true THEN 1 ELSE 0 END) as successful_count,
                    AVG(total_events) as avg_events,
                    AVG(error_count) as avg_errors
                FROM conversation_metadata
                WHERE start_timestamp >= ?
                GROUP BY agent_type
            """, [recent_date])
            
            for row in result.fetchall():
                agent_type, conv_count, success_count, avg_events, avg_errors = row
                stats['recent_analysis'][agent_type] = {
                    'conversations': conv_count,
                    'success_rate': success_count / conv_count if conv_count > 0 else 0,
                    'avg_events': avg_events,
                    'avg_errors': avg_errors
                }
            
            return stats
            
        except Exception as e:
            logger.error(f"Pattern detection summary failed: {e}")
            return {'error': str(e)}

# Create alias for backward compatibility
PatternDetectionEngine = ConversationPatternDetector