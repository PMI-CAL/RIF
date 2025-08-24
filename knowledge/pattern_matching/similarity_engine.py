"""
Similarity Engine - Advanced Pattern Matching Component

This module handles similarity detection between issues and patterns using multiple
techniques including semantic analysis, vector similarity, and structural comparison.

Key Features:
- Issue similarity detection with configurable thresholds
- Semantic similarity using NLP techniques
- Technology stack compatibility analysis
- Vector-based similarity search
- Multi-dimensional similarity scoring
"""

import re
import math
import logging
import numpy as np
from typing import Dict, List, Any, Optional, Set, Tuple
from datetime import datetime
from dataclasses import dataclass

# Import knowledge system interfaces
try:
    from knowledge.interface import get_knowledge_system
    from knowledge.database.database_interface import RIFDatabase
except ImportError:
    def get_knowledge_system():
        raise ImportError("Knowledge system not available")
    
    class RIFDatabase:
        pass

# Import pattern application core components
from knowledge.pattern_application.core import (
    Pattern, IssueContext, TechStack
)


@dataclass
class SimilarityResult:
    """Result of similarity analysis."""
    score: float
    factors: Dict[str, float]
    explanation: str
    confidence: float


class SimilarityEngine:
    """
    Advanced similarity detection engine for issues and patterns.
    
    This engine provides multiple similarity detection techniques:
    - Semantic similarity using text analysis
    - Structural similarity based on metadata
    - Technology stack compatibility
    - Vector-based similarity (when available)
    - Historical pattern matching
    """
    
    def __init__(self, knowledge_system=None, database: Optional[RIFDatabase] = None):
        """Initialize the similarity engine."""
        self.logger = logging.getLogger(__name__)
        self.knowledge_system = knowledge_system or get_knowledge_system()
        self.database = database
        
        # Initialize NLP components
        self._load_language_models()
        
        # Semantic keyword clusters for domain analysis
        self.semantic_clusters = {
            'authentication': ['auth', 'login', 'user', 'session', 'token', 'jwt', 'oauth', 'sso'],
            'database': ['db', 'sql', 'query', 'schema', 'migration', 'orm', 'model', 'table'],
            'api': ['rest', 'endpoint', 'service', 'http', 'request', 'response', 'json', 'api'],
            'frontend': ['ui', 'component', 'render', 'display', 'react', 'vue', 'angular', 'css'],
            'backend': ['server', 'service', 'microservice', 'processing', 'business', 'logic'],
            'testing': ['test', 'unit', 'integration', 'validation', 'mock', 'spec', 'coverage'],
            'deployment': ['deploy', 'ci/cd', 'docker', 'kubernetes', 'aws', 'cloud', 'infra'],
            'performance': ['optimize', 'cache', 'speed', 'latency', 'scalability', 'load'],
            'security': ['security', 'vulnerability', 'encryption', 'https', 'sanitize', 'xss'],
            'monitoring': ['logs', 'metrics', 'monitoring', 'alerts', 'observability', 'tracing']
        }
        
        # Technology compatibility matrix
        self.tech_compatibility = {
            'javascript': ['typescript', 'node.js', 'npm', 'yarn'],
            'python': ['django', 'flask', 'fastapi', 'pip', 'conda'],
            'java': ['spring', 'maven', 'gradle', 'junit'],
            'go': ['gin', 'echo', 'gorm', 'go mod'],
            'rust': ['tokio', 'actix', 'cargo', 'serde'],
            'c#': ['asp.net', '.net', 'nuget', 'entity framework']
        }
        
        self.logger.info("Similarity Engine initialized")
    
    def find_similar_issues(self, issue_context: IssueContext,
                          similarity_threshold: float = 0.7,
                          limit: int = 20) -> List[Dict[str, Any]]:
        """
        Find issues similar to the given context.
        
        This implements the core similarity detection required by Issue #76
        acceptance criteria: "Finds relevant similar issues"
        
        Args:
            issue_context: Context of the current issue
            similarity_threshold: Minimum similarity score (0.0 to 1.0)
            limit: Maximum number of similar issues to return
            
        Returns:
            List of similar issue data sorted by similarity score
        """
        self.logger.info(f"Finding similar issues for {issue_context.issue_id}")
        
        try:
            # Get all issues from knowledge system
            all_issues = self._get_all_issues()
            
            if not all_issues:
                self.logger.warning("No historical issues found")
                return []
            
            # Calculate similarity for each issue
            similar_issues = []
            for issue in all_issues:
                # Skip the same issue
                if issue.get('issue_id') == issue_context.issue_id:
                    continue
                
                # Calculate comprehensive similarity
                similarity_result = self.calculate_issue_similarity(issue_context, issue)
                
                if similarity_result.score >= similarity_threshold:
                    issue_data = issue.copy()
                    issue_data['similarity_score'] = similarity_result.score
                    issue_data['similarity_factors'] = similarity_result.factors
                    issue_data['similarity_explanation'] = similarity_result.explanation
                    similar_issues.append(issue_data)
            
            # Sort by similarity score (highest first)
            similar_issues.sort(key=lambda x: x['similarity_score'], reverse=True)
            
            result_issues = similar_issues[:limit]
            self.logger.info(f"Found {len(result_issues)} similar issues")
            
            return result_issues
            
        except Exception as e:
            self.logger.error(f"Error finding similar issues: {str(e)}")
            return []
    
    def calculate_issue_similarity(self, issue_context: IssueContext,
                                 historical_issue: Dict[str, Any]) -> SimilarityResult:
        """
        Calculate comprehensive similarity between current issue and historical issue.
        
        Args:
            issue_context: Current issue context
            historical_issue: Historical issue data
            
        Returns:
            SimilarityResult with detailed similarity analysis
        """
        factors = {}
        
        # Semantic similarity (40% weight)
        current_text = f"{issue_context.title} {issue_context.description}"
        historical_text = f"{historical_issue.get('title', '')} {historical_issue.get('description', '')}"
        factors['semantic'] = self.calculate_semantic_similarity(current_text, historical_text)
        
        # Technology stack similarity (25% weight)
        historical_tech = self._extract_tech_stack_from_issue(historical_issue)
        factors['technology'] = self.calculate_tech_compatibility(
            issue_context.tech_stack, historical_tech
        )
        
        # Complexity similarity (15% weight)
        historical_complexity = historical_issue.get('complexity', 'medium')
        factors['complexity'] = self._calculate_complexity_similarity(
            issue_context.complexity, historical_complexity
        )
        
        # Domain similarity (10% weight)
        historical_domain = historical_issue.get('domain', 'general')
        factors['domain'] = self._calculate_domain_similarity(
            issue_context.domain, historical_domain
        )
        
        # Label/tag similarity (10% weight)
        historical_labels = historical_issue.get('labels', [])
        factors['labels'] = self._calculate_label_similarity(
            issue_context.labels, historical_labels
        )
        
        # Calculate weighted overall score
        weights = {
            'semantic': 0.40,
            'technology': 0.25,
            'complexity': 0.15,
            'domain': 0.10,
            'labels': 0.10
        }
        
        overall_score = sum(factors[key] * weights[key] for key in factors)
        
        # Calculate confidence based on data completeness
        confidence = self._calculate_similarity_confidence(issue_context, historical_issue)
        
        # Generate explanation
        explanation = self._generate_similarity_explanation(factors, weights)
        
        return SimilarityResult(
            score=overall_score,
            factors=factors,
            explanation=explanation,
            confidence=confidence
        )
    
    def calculate_semantic_similarity(self, text1: str, text2: str) -> float:
        """
        Calculate semantic similarity between two texts using advanced techniques.
        
        Args:
            text1: First text to compare
            text2: Second text to compare
            
        Returns:
            Similarity score (0.0 to 1.0)
        """
        if not text1 or not text2:
            return 0.0
        
        # Clean and tokenize texts
        tokens1 = self._tokenize_text(text1.lower())
        tokens2 = self._tokenize_text(text2.lower())
        
        if not tokens1 or not tokens2:
            return 0.0
        
        # Calculate multiple similarity measures
        scores = []
        
        # 1. Jaccard similarity (word overlap)
        jaccard_score = self._calculate_jaccard_similarity(tokens1, tokens2)
        scores.append(jaccard_score)
        
        # 2. Cosine similarity (TF-IDF based)
        cosine_score = self._calculate_cosine_similarity(tokens1, tokens2)
        scores.append(cosine_score)
        
        # 3. Semantic cluster similarity
        cluster_score = self._calculate_semantic_cluster_similarity(tokens1, tokens2)
        scores.append(cluster_score)
        
        # 4. N-gram similarity
        ngram_score = self._calculate_ngram_similarity(text1, text2)
        scores.append(ngram_score)
        
        # Combine scores with weights
        weights = [0.3, 0.3, 0.2, 0.2]
        final_score = sum(score * weight for score, weight in zip(scores, weights))
        
        return min(1.0, max(0.0, final_score))
    
    def calculate_tech_compatibility(self, tech1: Optional[TechStack],
                                   tech2: Optional[TechStack]) -> float:
        """
        Calculate technology stack compatibility score.
        
        Args:
            tech1: First technology stack
            tech2: Second technology stack
            
        Returns:
            Compatibility score (0.0 to 1.0)
        """
        if not tech1 or not tech2:
            return 0.5  # Neutral score when no tech info available
        
        compatibility_score = 0.0
        
        # Primary language compatibility (50% weight)
        lang_score = self._calculate_language_compatibility(
            tech1.primary_language, tech2.primary_language
        )
        compatibility_score += lang_score * 0.5
        
        # Framework compatibility (30% weight)
        framework_score = self._calculate_framework_compatibility(
            tech1.frameworks, tech2.frameworks, tech1.primary_language
        )
        compatibility_score += framework_score * 0.3
        
        # Database compatibility (10% weight)
        db_score = self._calculate_database_compatibility(
            tech1.databases, tech2.databases
        )
        compatibility_score += db_score * 0.1
        
        # Architecture pattern compatibility (10% weight)
        arch_score = self._calculate_architecture_compatibility(
            tech1.architecture_pattern, tech2.architecture_pattern
        )
        compatibility_score += arch_score * 0.1
        
        return compatibility_score
    
    def _get_all_issues(self) -> List[Dict[str, Any]]:
        """Get all historical issues from the knowledge system."""
        try:
            # Query knowledge system for all issues
            results = self.knowledge_system.retrieve_knowledge(
                query="*",
                collection="issues",
                n_results=1000  # Get many issues for similarity comparison
            )
            
            issues = []
            for result in results:
                try:
                    issue_data = self._convert_result_to_issue(result)
                    issues.append(issue_data)
                except Exception as e:
                    self.logger.debug(f"Failed to convert result to issue: {str(e)}")
                    continue
            
            return issues
            
        except Exception as e:
            self.logger.error(f"Failed to retrieve historical issues: {str(e)}")
            return []
    
    def _convert_result_to_issue(self, result: Dict[str, Any]) -> Dict[str, Any]:
        """Convert knowledge system result to issue data."""
        content = result.get('content', {})
        if isinstance(content, str):
            import json
            try:
                content = json.loads(content)
            except json.JSONDecodeError:
                content = {"description": content}
        
        return {
            'issue_id': content.get('issue_id', result.get('id', '')),
            'title': content.get('title', ''),
            'description': content.get('description', ''),
            'complexity': content.get('complexity', 'medium'),
            'domain': content.get('domain', 'general'),
            'labels': content.get('labels', []),
            'tech_stack': content.get('tech_stack', {}),
            'state': content.get('state', 'unknown'),
            'outcome': content.get('outcome', 'unknown')
        }
    
    def _extract_tech_stack_from_issue(self, issue: Dict[str, Any]) -> Optional[TechStack]:
        """Extract TechStack object from issue data."""
        tech_data = issue.get('tech_stack', {})
        if not tech_data:
            return None
        
        return TechStack(
            primary_language=tech_data.get('primary_language', ''),
            frameworks=tech_data.get('frameworks', []),
            databases=tech_data.get('databases', []),
            tools=tech_data.get('tools', []),
            architecture_pattern=tech_data.get('architecture_pattern'),
            deployment_target=tech_data.get('deployment_target')
        )
    
    def _calculate_complexity_similarity(self, complexity1: str, complexity2: str) -> float:
        """Calculate similarity between complexity levels."""
        complexity_levels = ['low', 'medium', 'high', 'very-high']
        
        try:
            idx1 = complexity_levels.index(complexity1)
            idx2 = complexity_levels.index(complexity2)
            
            # Perfect match
            if idx1 == idx2:
                return 1.0
            
            # Adjacent levels
            diff = abs(idx1 - idx2)
            if diff == 1:
                return 0.7
            elif diff == 2:
                return 0.4
            else:
                return 0.1
                
        except ValueError:
            return 0.5  # Unknown complexity
    
    def _calculate_domain_similarity(self, domain1: str, domain2: str) -> float:
        """Calculate similarity between domains."""
        if domain1 == domain2:
            return 1.0
        elif domain1 == 'general' or domain2 == 'general':
            return 0.5
        else:
            return 0.0
    
    def _calculate_label_similarity(self, labels1: List[str], labels2: List[str]) -> float:
        """Calculate similarity between label sets."""
        if not labels1 or not labels2:
            return 0.0
        
        set1 = set(label.lower() for label in labels1)
        set2 = set(label.lower() for label in labels2)
        
        intersection = len(set1.intersection(set2))
        union = len(set1.union(set2))
        
        return intersection / union if union > 0 else 0.0
    
    def _tokenize_text(self, text: str) -> List[str]:
        """Tokenize text into meaningful words."""
        # Remove special characters and split into words
        words = re.findall(r'\b\w+\b', text.lower())
        
        # Filter out very short words and common stop words
        stop_words = {'the', 'a', 'an', 'and', 'or', 'but', 'in', 'on', 'at', 'to', 'for', 'of', 'with', 'by', 'is', 'are', 'was', 'were', 'be', 'been', 'have', 'has', 'had', 'do', 'does', 'did', 'will', 'would', 'should', 'could'}
        filtered_words = [word for word in words if len(word) > 2 and word not in stop_words]
        
        return filtered_words
    
    def _calculate_jaccard_similarity(self, tokens1: List[str], tokens2: List[str]) -> float:
        """Calculate Jaccard similarity coefficient."""
        set1 = set(tokens1)
        set2 = set(tokens2)
        
        intersection = len(set1.intersection(set2))
        union = len(set1.union(set2))
        
        return intersection / union if union > 0 else 0.0
    
    def _calculate_cosine_similarity(self, tokens1: List[str], tokens2: List[str]) -> float:
        """Calculate cosine similarity using TF-IDF-like weighting."""
        # Create vocabulary
        vocab = list(set(tokens1 + tokens2))
        
        if not vocab:
            return 0.0
        
        # Create term frequency vectors
        def create_tf_vector(tokens, vocab):
            tf_vector = []
            for term in vocab:
                tf = tokens.count(term) / len(tokens) if tokens else 0
                tf_vector.append(tf)
            return np.array(tf_vector)
        
        vec1 = create_tf_vector(tokens1, vocab)
        vec2 = create_tf_vector(tokens2, vocab)
        
        # Calculate cosine similarity
        dot_product = np.dot(vec1, vec2)
        norm1 = np.linalg.norm(vec1)
        norm2 = np.linalg.norm(vec2)
        
        if norm1 == 0 or norm2 == 0:
            return 0.0
        
        return dot_product / (norm1 * norm2)
    
    def _calculate_semantic_cluster_similarity(self, tokens1: List[str], tokens2: List[str]) -> float:
        """Calculate similarity based on semantic clusters."""
        clusters1 = self._identify_semantic_clusters(tokens1)
        clusters2 = self._identify_semantic_clusters(tokens2)
        
        if not clusters1 or not clusters2:
            return 0.0
        
        # Calculate cluster overlap
        common_clusters = clusters1.intersection(clusters2)
        total_clusters = clusters1.union(clusters2)
        
        return len(common_clusters) / len(total_clusters) if total_clusters else 0.0
    
    def _calculate_ngram_similarity(self, text1: str, text2: str, n: int = 2) -> float:
        """Calculate n-gram similarity."""
        def get_ngrams(text, n):
            # Clean text
            text = re.sub(r'[^a-zA-Z0-9\s]', '', text.lower())
            words = text.split()
            ngrams = []
            for i in range(len(words) - n + 1):
                ngrams.append(' '.join(words[i:i+n]))
            return set(ngrams)
        
        ngrams1 = get_ngrams(text1, n)
        ngrams2 = get_ngrams(text2, n)
        
        if not ngrams1 or not ngrams2:
            return 0.0
        
        intersection = len(ngrams1.intersection(ngrams2))
        union = len(ngrams1.union(ngrams2))
        
        return intersection / union if union > 0 else 0.0
    
    def _identify_semantic_clusters(self, tokens: List[str]) -> Set[str]:
        """Identify which semantic clusters the tokens belong to."""
        identified_clusters = set()
        
        token_set = set(tokens)
        
        for cluster_name, cluster_keywords in self.semantic_clusters.items():
            # Check if any cluster keywords are present
            if any(keyword in token_set for keyword in cluster_keywords):
                identified_clusters.add(cluster_name)
        
        return identified_clusters
    
    def _calculate_language_compatibility(self, lang1: str, lang2: str) -> float:
        """Calculate compatibility between programming languages."""
        if not lang1 or not lang2:
            return 0.5
        
        lang1 = lang1.lower()
        lang2 = lang2.lower()
        
        # Exact match
        if lang1 == lang2:
            return 1.0
        
        # Check compatibility matrix
        for base_lang, compatible_langs in self.tech_compatibility.items():
            if lang1 == base_lang and lang2 in compatible_langs:
                return 0.8
            if lang2 == base_lang and lang1 in compatible_langs:
                return 0.8
        
        # Related languages
        related_groups = [
            ['javascript', 'typescript'],
            ['python', 'python3'],
            ['c', 'c++'],
            ['java', 'scala', 'kotlin'],
            ['c#', 'f#', 'vb.net']
        ]
        
        for group in related_groups:
            if lang1 in group and lang2 in group:
                return 0.6
        
        return 0.0
    
    def _calculate_framework_compatibility(self, frameworks1: List[str], 
                                         frameworks2: List[str], 
                                         primary_language: str) -> float:
        """Calculate framework compatibility."""
        if not frameworks1 or not frameworks2:
            return 0.5
        
        set1 = set(fw.lower() for fw in frameworks1)
        set2 = set(fw.lower() for fw in frameworks2)
        
        # Direct overlap
        overlap = len(set1.intersection(set2))
        total = len(set1.union(set2))
        
        if overlap > 0:
            return overlap / total
        
        # Check for compatible frameworks within same language ecosystem
        if primary_language:
            compatible_frameworks = self.tech_compatibility.get(primary_language.lower(), [])
            compatible_set = set(fw.lower() for fw in compatible_frameworks)
            
            if set1.intersection(compatible_set) and set2.intersection(compatible_set):
                return 0.5
        
        return 0.0
    
    def _calculate_database_compatibility(self, databases1: List[str], 
                                        databases2: List[str]) -> float:
        """Calculate database compatibility."""
        if not databases1 or not databases2:
            return 0.5
        
        set1 = set(db.lower() for db in databases1)
        set2 = set(db.lower() for db in databases2)
        
        # Direct overlap
        overlap = len(set1.intersection(set2))
        total = len(set1.union(set2))
        
        if overlap > 0:
            return overlap / total
        
        # Similar database types
        sql_databases = {'mysql', 'postgresql', 'sqlite', 'mariadb', 'mssql', 'oracle'}
        nosql_databases = {'mongodb', 'redis', 'cassandra', 'dynamodb', 'couchdb'}
        
        set1_sql = bool(set1.intersection(sql_databases))
        set2_sql = bool(set2.intersection(sql_databases))
        set1_nosql = bool(set1.intersection(nosql_databases))
        set2_nosql = bool(set2.intersection(nosql_databases))
        
        if (set1_sql and set2_sql) or (set1_nosql and set2_nosql):
            return 0.3
        
        return 0.0
    
    def _calculate_architecture_compatibility(self, arch1: Optional[str], 
                                            arch2: Optional[str]) -> float:
        """Calculate architecture pattern compatibility."""
        if not arch1 or not arch2:
            return 0.5
        
        if arch1.lower() == arch2.lower():
            return 1.0
        
        # Compatible architecture patterns
        compatible_patterns = [
            ['mvc', 'mvp', 'mvvm'],
            ['microservices', 'soa'],
            ['rest', 'restful'],
            ['layered', 'n-tier']
        ]
        
        arch1_lower = arch1.lower()
        arch2_lower = arch2.lower()
        
        for pattern_group in compatible_patterns:
            if arch1_lower in pattern_group and arch2_lower in pattern_group:
                return 0.7
        
        return 0.0
    
    def _calculate_similarity_confidence(self, issue_context: IssueContext,
                                       historical_issue: Dict[str, Any]) -> float:
        """Calculate confidence in similarity score based on data completeness."""
        confidence_factors = []
        
        # Title completeness
        if issue_context.title and historical_issue.get('title'):
            confidence_factors.append(1.0)
        else:
            confidence_factors.append(0.0)
        
        # Description completeness
        if issue_context.description and historical_issue.get('description'):
            confidence_factors.append(1.0)
        else:
            confidence_factors.append(0.5)
        
        # Technology stack completeness
        if issue_context.tech_stack and historical_issue.get('tech_stack'):
            confidence_factors.append(1.0)
        else:
            confidence_factors.append(0.3)
        
        # Labels completeness
        if issue_context.labels and historical_issue.get('labels'):
            confidence_factors.append(0.8)
        else:
            confidence_factors.append(0.0)
        
        return sum(confidence_factors) / len(confidence_factors)
    
    def _generate_similarity_explanation(self, factors: Dict[str, float], 
                                       weights: Dict[str, float]) -> str:
        """Generate human-readable explanation of similarity score."""
        explanations = []
        
        for factor, score in factors.items():
            weight = weights.get(factor, 0.0)
            contribution = score * weight
            
            if score > 0.7:
                level = "high"
            elif score > 0.4:
                level = "moderate"
            else:
                level = "low"
            
            explanations.append(f"{factor} similarity: {level} ({score:.2f}, weight: {weight:.0%})")
        
        return "; ".join(explanations)
    
    def _load_language_models(self):
        """Load language models and NLP components."""
        # This would load more sophisticated NLP models if available
        # For now, using rule-based approaches
        self.logger.debug("Language models loaded (rule-based implementation)")