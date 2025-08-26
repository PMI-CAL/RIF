#!/usr/bin/env python3
"""
Project Knowledge Initializer for RIF

This script initializes a clean knowledge base for a new project by:
1. Creating minimal knowledge structure
2. Loading preserved patterns from cleanup
3. Setting up project-specific configuration
4. Initializing empty databases and indexes

Usage:
    python scripts/init_project_knowledge.py --project-name "My Project" --project-type "web-app"
"""

import json
import argparse
import sqlite3
from pathlib import Path
from datetime import datetime
from typing import Dict, List, Optional
import logging

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


class ProjectKnowledgeInitializer:
    """Initializes clean knowledge base for new projects"""
    
    def __init__(self, knowledge_dir: str = "knowledge"):
        self.knowledge_dir = Path(knowledge_dir)
        self.timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    
    def initialize_project_knowledge(
        self, 
        project_name: str, 
        project_type: str,
        preserved_knowledge_file: Optional[str] = None
    ) -> Dict:
        """Initialize clean knowledge base for new project"""
        logger.info(f"Initializing knowledge base for project: {project_name}")
        
        # Create base knowledge structure
        self._create_knowledge_structure()
        
        # Initialize project metadata
        project_info = {
            'project': {
                'name': project_name,
                'type': project_type,
                'initialized': datetime.now().isoformat(),
                'rif_version': '2.0',
                'knowledge_schema_version': '1.0'
            },
            'configuration': {
                'quality_thresholds': self._get_default_quality_thresholds(project_type),
                'agent_preferences': self._get_default_agent_preferences(project_type),
                'workflow_settings': self._get_default_workflow_settings(project_type)
            },
            'statistics': {
                'patterns_loaded': 0,
                'decisions_loaded': 0,
                'learnings_loaded': 0,
                'initialization_date': datetime.now().isoformat()
            }
        }
        
        # Load preserved knowledge if available
        if preserved_knowledge_file and Path(preserved_knowledge_file).exists():
            stats = self._load_preserved_knowledge(preserved_knowledge_file)
            project_info['statistics'].update(stats)
        
        # Initialize core directories with minimal content
        self._initialize_core_directories(project_info)
        
        # Initialize databases
        self._initialize_databases(project_name)
        
        # Create initial configuration files
        self._create_configuration_files(project_info)
        
        logger.info("Project knowledge base initialization complete")
        return project_info
    
    def _create_knowledge_structure(self):
        """Create minimal knowledge directory structure"""
        logger.info("Creating knowledge directory structure...")
        
        # Essential directories for any RIF project
        essential_dirs = [
            'patterns',      # Reusable code/architecture patterns
            'decisions',     # Project architecture decisions
            'learning',      # Learning and improvements
            'context',       # Context optimization components
            'conversations', # Conversation storage system
            'embeddings',    # Embedding generation
            'parsing',       # Code parsing utilities
            'integration',   # System integration components
            'quality_metrics', # Quality measurement
            'plans',         # Planning artifacts
            'validation',    # Validation results
            'chromadb'       # Vector database
        ]
        
        for dirname in essential_dirs:
            dir_path = self.knowledge_dir / dirname
            dir_path.mkdir(parents=True, exist_ok=True)
            
            # Create README for empty directories
            readme_path = dir_path / "README.md"
            if not readme_path.exists():
                with open(readme_path, 'w') as f:
                    f.write(f"# {dirname.title()}\n\nThis directory will contain {dirname.replace('_', ' ')} for the project.\n")
    
    def _get_default_quality_thresholds(self, project_type: str) -> Dict:
        """Get default quality thresholds based on project type"""
        base_thresholds = {
            'test_coverage': 80.0,
            'code_quality_score': 7.5,
            'documentation_coverage': 70.0,
            'security_score': 8.0,
            'performance_threshold': 1000  # ms
        }
        
        # Adjust based on project type
        if project_type in ['enterprise', 'financial', 'healthcare']:
            base_thresholds.update({
                'test_coverage': 95.0,
                'security_score': 9.0,
                'documentation_coverage': 90.0
            })
        elif project_type in ['prototype', 'poc', 'experimental']:
            base_thresholds.update({
                'test_coverage': 60.0,
                'code_quality_score': 6.0,
                'documentation_coverage': 50.0
            })
        
        return base_thresholds
    
    def _get_default_agent_preferences(self, project_type: str) -> Dict:
        """Get default agent preferences based on project type"""
        return {
            'preferred_complexity_threshold': 'medium',
            'enable_parallel_processing': True,
            'auto_quality_gates': True,
            'learning_mode': 'active',
            'pattern_application': 'automatic',
            'context_optimization': True
        }
    
    def _get_default_workflow_settings(self, project_type: str) -> Dict:
        """Get default workflow settings based on project type"""
        return {
            'state_machine_enabled': True,
            'automatic_transitions': True,
            'checkpoint_frequency': 'high',
            'validation_strictness': 'standard',
            'parallel_agent_limit': 4
        }
    
    def _load_preserved_knowledge(self, preserved_file: str) -> Dict:
        """Load preserved knowledge from cleanup export"""
        logger.info(f"Loading preserved knowledge from {preserved_file}")
        
        try:
            with open(preserved_file, 'r') as f:
                preserved_data = json.load(f)
            
            stats = {
                'patterns_loaded': 0,
                'decisions_loaded': 0, 
                'learnings_loaded': 0
            }
            
            # Load patterns
            patterns_dir = self.knowledge_dir / 'patterns'
            for pattern in preserved_data.get('patterns', []):
                pattern_file = patterns_dir / f"preserved_{pattern.get('source_file', 'unknown.json')}"
                with open(pattern_file, 'w') as f:
                    json.dump(pattern, f, indent=2)
                stats['patterns_loaded'] += 1
            
            # Load decisions
            decisions_dir = self.knowledge_dir / 'decisions'
            for decision in preserved_data.get('decisions', []):
                decision_file = decisions_dir / f"preserved_{decision.get('source_file', 'unknown.json')}"
                with open(decision_file, 'w') as f:
                    json.dump(decision, f, indent=2)
                stats['decisions_loaded'] += 1
            
            # Load learnings
            learning_dir = self.knowledge_dir / 'learning'
            for learning in preserved_data.get('learnings', []):
                learning_file = learning_dir / f"preserved_{learning.get('source_file', 'unknown.json')}"
                with open(learning_file, 'w') as f:
                    json.dump(learning, f, indent=2)
                stats['learnings_loaded'] += 1
            
            logger.info(f"Loaded {stats['patterns_loaded']} patterns, {stats['decisions_loaded']} decisions, {stats['learnings_loaded']} learnings")
            return stats
            
        except Exception as e:
            logger.error(f"Could not load preserved knowledge: {e}")
            return {'patterns_loaded': 0, 'decisions_loaded': 0, 'learnings_loaded': 0}
    
    def _initialize_core_directories(self, project_info: Dict):
        """Initialize core directories with minimal required files"""
        
        # Create __init__.py files for Python directories
        python_dirs = ['context', 'conversations', 'embeddings', 'parsing', 'integration']
        for dirname in python_dirs:
            init_file = self.knowledge_dir / dirname / '__init__.py'
            if not init_file.exists():
                with open(init_file, 'w') as f:
                    f.write(f'"""{dirname.title()} module for {project_info["project"]["name"]}"""\n')
        
        # Create initial patterns file
        patterns_index = self.knowledge_dir / 'patterns' / 'index.json'
        if not patterns_index.exists():
            with open(patterns_index, 'w') as f:
                json.dump({
                    'project': project_info['project']['name'],
                    'patterns': [],
                    'last_updated': datetime.now().isoformat()
                }, f, indent=2)
        
        # Create initial decisions log
        decisions_log = self.knowledge_dir / 'decisions' / 'decision_log.json'
        if not decisions_log.exists():
            with open(decisions_log, 'w') as f:
                json.dump({
                    'project': project_info['project']['name'],
                    'decisions': [],
                    'last_updated': datetime.now().isoformat()
                }, f, indent=2)
    
    def _initialize_databases(self, project_name: str):
        """Initialize empty databases for the project"""
        logger.info("Initializing databases...")
        
        # Initialize conversations database
        conversations_db = self.knowledge_dir / 'conversations.duckdb'
        if not conversations_db.exists():
            # Create empty DuckDB file
            conversations_db.touch()
        
        # Initialize orchestration database  
        orchestration_db = self.knowledge_dir / 'orchestration.duckdb'
        if not orchestration_db.exists():
            orchestration_db.touch()
        
        # Initialize ChromaDB directory
        chromadb_dir = self.knowledge_dir / 'chromadb'
        if not (chromadb_dir / 'chroma.sqlite3').exists():
            # Create minimal ChromaDB structure
            chroma_sqlite = chromadb_dir / 'chroma.sqlite3'
            chroma_sqlite.touch()
    
    def _create_configuration_files(self, project_info: Dict):
        """Create initial configuration files"""
        logger.info("Creating configuration files...")
        
        # Create project metadata file
        metadata_file = self.knowledge_dir / 'project_metadata.json'
        with open(metadata_file, 'w') as f:
            json.dump(project_info, f, indent=2)
        
        # Create migration guide
        migration_guide = self.knowledge_dir / 'MIGRATION_GUIDE.md'
        if not migration_guide.exists():
            guide_content = f"""# Migration Guide - {project_info['project']['name']}

This knowledge base was initialized on {project_info['project']['initialized']} using RIF v{project_info['project']['rif_version']}.

## Project Configuration

- **Project Type**: {project_info['project']['type']}
- **Quality Thresholds**: See project_metadata.json
- **Preserved Knowledge**: {project_info['statistics']['patterns_loaded']} patterns, {project_info['statistics']['decisions_loaded']} decisions

## Directory Structure

- `patterns/` - Reusable code and architecture patterns
- `decisions/` - Project architecture decisions  
- `learning/` - Learning and improvement records
- `context/` - Context optimization components
- `conversations/` - Conversation storage system
- `quality_metrics/` - Quality measurement framework

## Next Steps

1. Configure project-specific quality thresholds in project_metadata.json
2. Add project-specific patterns to patterns/ directory
3. Begin development using RIF orchestration
4. Monitor learning/ directory for continuous improvements

## Rollback

If you need to restore from the original knowledge base, use the rollback script provided during cleanup.
"""
            with open(migration_guide, 'w') as f:
                f.write(guide_content)
    
    def generate_initialization_report(self, project_info: Dict) -> str:
        """Generate initialization report"""
        return f"""# Project Knowledge Initialization Report

## Project Details
- **Name**: {project_info['project']['name']}
- **Type**: {project_info['project']['type']}  
- **Initialized**: {project_info['project']['initialized']}
- **RIF Version**: {project_info['project']['rif_version']}

## Knowledge Base Statistics
- **Patterns Loaded**: {project_info['statistics']['patterns_loaded']}
- **Decisions Loaded**: {project_info['statistics']['decisions_loaded']}
- **Learnings Loaded**: {project_info['statistics']['learnings_loaded']}

## Quality Configuration
- **Test Coverage Threshold**: {project_info['configuration']['quality_thresholds']['test_coverage']}%
- **Code Quality Score**: {project_info['configuration']['quality_thresholds']['code_quality_score']}/10
- **Security Score**: {project_info['configuration']['quality_thresholds']['security_score']}/10

## Directory Structure Created
✅ Essential knowledge directories
✅ Database files initialized
✅ Configuration files created
✅ Migration guide generated

## Next Steps
1. Customize quality thresholds in project_metadata.json
2. Add project-specific patterns
3. Begin development with `rif-init.sh`
4. Use GitHub issues to trigger RIF orchestration

The knowledge base is ready for development!
"""


def main():
    """Main execution function"""
    parser = argparse.ArgumentParser(description="Initialize clean knowledge base for new project")
    parser.add_argument(
        "--project-name",
        required=True,
        help="Name of the project"
    )
    parser.add_argument(
        "--project-type", 
        required=True,
        choices=['web-app', 'mobile-app', 'desktop-app', 'library', 'framework', 
                'api', 'microservices', 'enterprise', 'prototype', 'poc', 'experimental'],
        help="Type of project"
    )
    parser.add_argument(
        "--knowledge-dir",
        default="knowledge",
        help="Path to knowledge directory (default: knowledge)"
    )
    parser.add_argument(
        "--preserved-knowledge",
        help="Path to preserved knowledge JSON file from cleanup"
    )
    parser.add_argument(
        "--output-report",
        help="Path to save initialization report"
    )
    
    args = parser.parse_args()
    
    # Initialize the project knowledge base
    initializer = ProjectKnowledgeInitializer(args.knowledge_dir)
    
    project_info = initializer.initialize_project_knowledge(
        project_name=args.project_name,
        project_type=args.project_type,
        preserved_knowledge_file=args.preserved_knowledge
    )
    
    # Generate report
    report = initializer.generate_initialization_report(project_info)
    
    # Save report if requested
    if args.output_report:
        with open(args.output_report, 'w') as f:
            f.write(report)
        logger.info(f"Initialization report saved to: {args.output_report}")
    
    # Print summary
    print("\n" + "="*60)
    print("PROJECT KNOWLEDGE INITIALIZATION COMPLETE")
    print("="*60)
    print(report)


if __name__ == "__main__":
    main()