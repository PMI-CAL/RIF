"""
Requirement Detector

Analyzes project context and detects MCP server requirements
based on technology stack, capabilities needed, and agent types.

Issue: #82 - Implement dynamic MCP loader
Component: Requirement detection system
"""

import os
import json
import logging
import asyncio
from typing import Set, Dict, Any, List
from pathlib import Path

logger = logging.getLogger(__name__)


class RequirementDetector:
    """
    Detects MCP server requirements from project context analysis
    
    Analyzes:
    - Technology stack (languages, frameworks, tools)
    - Project structure and files
    - Integration needs (GitHub, databases, cloud)
    - Agent capabilities required
    - Complexity and performance requirements
    """
    
    def __init__(self):
        """Initialize the requirement detector"""
        self.technology_patterns = {
            'javascript': ['.js', '.ts', '.jsx', '.tsx', 'package.json'],
            'nodejs': ['package.json', 'node_modules/', '.npmrc'],
            'python': ['.py', 'requirements.txt', 'setup.py', 'pyproject.toml', '__pycache__/'],
            'java': ['.java', '.class', 'pom.xml', 'build.gradle', '.mvn/'],
            'go': ['.go', 'go.mod', 'go.sum'],
            'rust': ['.rs', 'Cargo.toml', 'Cargo.lock', 'target/'],
            'docker': ['Dockerfile', 'docker-compose.yml', '.dockerignore'],
            'git': ['.git/', '.gitignore', '.gitattributes'],
            'github': ['.github/', '.github/workflows/']
        }
        
        self.integration_patterns = {
            'database': ['migrations/', 'schema.sql', '.env', 'database.yml'],
            'cloud': ['terraform/', '.aws/', '.azure/', 'cloud-config'],
            'api': ['api/', 'routes/', 'endpoints/', 'swagger.json', 'openapi.yml'],
            'testing': ['test/', 'tests/', 'spec/', '__tests__/', '.test.', '.spec.']
        }
        
        self.complexity_indicators = {
            'high': ['microservices', 'kubernetes', 'terraform', 'complex-architecture'],
            'medium': ['multiple-services', 'api-gateway', 'database-migrations'],
            'low': ['single-service', 'simple-structure']
        }
    
    async def detect_technology_stack(self, project_path: str) -> Set[str]:
        """
        Detect technology stack from project structure
        
        Args:
            project_path: Path to project directory
            
        Returns:
            Set of detected technologies
        """
        detected = set()
        
        try:
            project_root = Path(project_path)
            if not project_root.exists():
                logger.warning(f"Project path does not exist: {project_path}")
                return detected
            
            # Scan project files
            files_and_dirs = self._scan_project_structure(project_root)
            
            # Check technology patterns
            for tech, patterns in self.technology_patterns.items():
                for pattern in patterns:
                    if any(pattern in item for item in files_and_dirs):
                        detected.add(tech)
                        logger.debug(f"Detected {tech} from pattern {pattern}")
            
            # Special detection logic
            await self._detect_special_technologies(project_root, detected)
            
            logger.info(f"Detected technology stack: {detected}")
            return detected
            
        except Exception as e:
            logger.error(f"Failed to detect technology stack: {e}")
            return detected
    
    async def detect_integration_needs(self, project_path: str) -> Set[str]:
        """
        Detect integration requirements (databases, cloud, APIs)
        
        Args:
            project_path: Path to project directory
            
        Returns:
            Set of integration requirements
        """
        integrations = set()
        
        try:
            project_root = Path(project_path)
            files_and_dirs = self._scan_project_structure(project_root)
            
            # Check integration patterns
            for integration, patterns in self.integration_patterns.items():
                for pattern in patterns:
                    if any(pattern in item for item in files_and_dirs):
                        integrations.add(integration)
            
            # Check for specific service integrations
            await self._detect_service_integrations(project_root, integrations)
            
            logger.info(f"Detected integrations: {integrations}")
            return integrations
            
        except Exception as e:
            logger.error(f"Failed to detect integrations: {e}")
            return integrations
    
    async def assess_complexity(self, project_path: str, technology_stack: Set[str]) -> str:
        """
        Assess project complexity level
        
        Args:
            project_path: Path to project directory
            technology_stack: Detected technology stack
            
        Returns:
            Complexity level: low, medium, high, very-high
        """
        try:
            project_root = Path(project_path)
            
            # Count indicators
            complexity_score = 0
            
            # File count scoring
            file_count = len(list(project_root.rglob('*'))) if project_root.exists() else 0
            if file_count > 1000:
                complexity_score += 3
            elif file_count > 500:
                complexity_score += 2
            elif file_count > 100:
                complexity_score += 1
            
            # Technology diversity scoring
            if len(technology_stack) > 5:
                complexity_score += 2
            elif len(technology_stack) > 3:
                complexity_score += 1
            
            # Architecture indicators
            if project_root.exists():
                files_content = await self._sample_file_content(project_root)
                
                if any(keyword in files_content.lower() for keyword in 
                       ['microservice', 'kubernetes', 'terraform', 'distributed']):
                    complexity_score += 3
                elif any(keyword in files_content.lower() for keyword in 
                         ['service', 'api', 'database', 'cloud']):
                    complexity_score += 1
            
            # Determine complexity level
            if complexity_score >= 6:
                return 'very-high'
            elif complexity_score >= 4:
                return 'high'
            elif complexity_score >= 2:
                return 'medium'
            else:
                return 'low'
                
        except Exception as e:
            logger.error(f"Failed to assess complexity: {e}")
            return 'medium'  # Safe default
    
    def _scan_project_structure(self, project_root: Path, max_depth: int = 3) -> List[str]:
        """
        Scan project structure up to specified depth
        
        Args:
            project_root: Project root directory
            max_depth: Maximum scanning depth
            
        Returns:
            List of file and directory names
        """
        items = []
        
        try:
            for item in project_root.rglob('*'):
                # Check depth
                relative_parts = item.relative_to(project_root).parts
                if len(relative_parts) > max_depth:
                    continue
                
                items.append(str(item.relative_to(project_root)))
                
                # Limit scan for performance
                if len(items) > 10000:
                    logger.warning("Project scan limit reached, stopping at 10000 items")
                    break
                    
        except Exception as e:
            logger.error(f"Error scanning project structure: {e}")
        
        return items
    
    async def _detect_special_technologies(self, project_root: Path, detected: Set[str]):
        """
        Detect technologies requiring special logic
        
        Args:
            project_root: Project root directory
            detected: Set to add detected technologies to
        """
        try:
            # Check package.json for Node.js details
            package_json = project_root / 'package.json'
            if package_json.exists():
                content = json.loads(package_json.read_text())
                dependencies = {**content.get('dependencies', {}), 
                               **content.get('devDependencies', {})}
                
                if 'react' in dependencies:
                    detected.add('react')
                if 'vue' in dependencies:
                    detected.add('vue')
                if 'angular' in dependencies or '@angular/core' in dependencies:
                    detected.add('angular')
                if 'express' in dependencies:
                    detected.add('express')
                if 'typescript' in dependencies or any('.ts' in f for f in 
                    self._scan_project_structure(project_root)):
                    detected.add('typescript')
            
            # Check Python requirements
            requirements_file = project_root / 'requirements.txt'
            if requirements_file.exists():
                content = requirements_file.read_text()
                if 'django' in content.lower():
                    detected.add('django')
                if 'flask' in content.lower():
                    detected.add('flask')
                if 'fastapi' in content.lower():
                    detected.add('fastapi')
            
        except Exception as e:
            logger.debug(f"Error in special technology detection: {e}")
    
    async def _detect_service_integrations(self, project_root: Path, integrations: Set[str]):
        """
        Detect specific service integrations
        
        Args:
            project_root: Project root directory
            integrations: Set to add integrations to
        """
        try:
            # Check for GitHub integration
            if (project_root / '.github').exists():
                integrations.add('github')
            
            # Check for environment files indicating services
            env_files = ['.env', '.env.local', '.env.production']
            for env_file in env_files:
                env_path = project_root / env_file
                if env_path.exists():
                    content = env_path.read_text().upper()
                    if any(service in content for service in 
                           ['DATABASE_URL', 'POSTGRES', 'MYSQL', 'MONGODB']):
                        integrations.add('database')
                    if any(cloud in content for cloud in 
                           ['AWS_', 'AZURE_', 'GCP_', 'GOOGLE_CLOUD']):
                        integrations.add('cloud')
                    
        except Exception as e:
            logger.debug(f"Error in service integration detection: {e}")
    
    async def _sample_file_content(self, project_root: Path, max_files: int = 20) -> str:
        """
        Sample file content for keyword analysis
        
        Args:
            project_root: Project root directory
            max_files: Maximum files to sample
            
        Returns:
            Combined content sample
        """
        content_sample = ""
        file_count = 0
        
        try:
            # Focus on key files
            key_files = ['README.md', 'README.txt', 'package.json', 'requirements.txt', 
                        'setup.py', 'Cargo.toml', 'go.mod', 'pom.xml']
            
            for file_name in key_files:
                file_path = project_root / file_name
                if file_path.exists() and file_count < max_files:
                    try:
                        content_sample += file_path.read_text(encoding='utf-8')[:1000]
                        file_count += 1
                    except Exception:
                        continue
            
            # Sample additional files if needed
            if file_count < max_files:
                for file_path in project_root.rglob('*.md'):
                    if file_count >= max_files:
                        break
                    try:
                        content_sample += file_path.read_text(encoding='utf-8')[:500]
                        file_count += 1
                    except Exception:
                        continue
                        
        except Exception as e:
            logger.debug(f"Error sampling file content: {e}")
        
        return content_sample