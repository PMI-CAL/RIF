#!/usr/bin/env python3
"""
Knowledge Graph Auto-Update System

This module implements the knowledge graph update logic that processes
batched file changes from the FileChangeDetector and updates the knowledge
system accordingly.

This integrates with the hybrid knowledge system and provides intelligent
updates based on file change patterns.
"""

import os
import json
import time
import logging
from pathlib import Path
from typing import Dict, List, Any, Optional, Set
from dataclasses import dataclass
from datetime import datetime

# Import file change detection
from .file_change_detector import FileChangeDetector, FileChange

# Import knowledge system
try:
    from knowledge.interface import get_knowledge_system
    KNOWLEDGE_SYSTEM_AVAILABLE = True
except ImportError:
    KNOWLEDGE_SYSTEM_AVAILABLE = False


@dataclass
class UpdateContext:
    """Context for knowledge graph updates"""
    batch_id: str
    timestamp: datetime
    module: str
    change_count: int
    file_types: Set[str]
    update_type: str  # 'code', 'config', 'docs', 'tests'


class KnowledgeGraphUpdater:
    """
    Handles automated knowledge graph updates based on file changes.
    
    This class processes batched file changes and intelligently updates
    the knowledge system with relevant information about code changes,
    patterns, and architectural decisions.
    """
    
    def __init__(self, knowledge_path: str = "/Users/cal/DEV/RIF/knowledge"):
        self.logger = logging.getLogger(__name__)
        self.knowledge_path = Path(knowledge_path)
        
        # Initialize knowledge system
        self.knowledge_system = None
        if KNOWLEDGE_SYSTEM_AVAILABLE:
            try:
                self.knowledge_system = get_knowledge_system()
                self.logger.info("Knowledge system connected successfully")
            except Exception as e:
                self.logger.warning(f"Knowledge system not available: {e}")
        
        # Update statistics
        self.stats = {
            'batches_processed': 0,
            'files_analyzed': 0,
            'patterns_detected': 0,
            'updates_stored': 0,
            'last_update': None
        }
        
        self.logger.info("KnowledgeGraphUpdater initialized")
    
    def process_change_batch(self, module: str, changes: List[FileChange]) -> Optional[str]:
        """
        Process a batch of related file changes for knowledge graph updates.
        
        Args:
            module: Module/component name
            changes: List of file changes in this module
            
        Returns:
            Update context ID if successful, None otherwise
        """
        if not changes or not module:
            return None
            
        # Create update context
        context = self._create_update_context(module, changes)
        
        try:
            # Analyze changes for knowledge extraction
            knowledge_updates = self._analyze_changes_for_knowledge(context, changes)
            
            # Store updates in knowledge system
            if knowledge_updates and self.knowledge_system:
                update_id = self._store_knowledge_updates(context, knowledge_updates)
                self.stats['batches_processed'] += 1
                self.stats['files_analyzed'] += len(changes)
                self.stats['last_update'] = datetime.now().isoformat()
                return update_id
            else:
                self.logger.debug(f"No knowledge updates generated for {module}")
                return None
                
        except Exception as e:
            self.logger.error(f"Failed to process change batch for {module}: {e}")
            return None
    
    def _create_update_context(self, module: str, changes: List[FileChange]) -> UpdateContext:
        """Create update context from changes"""
        file_types = {Path(change.path).suffix for change in changes}
        
        # Determine update type based on files changed
        update_type = self._classify_update_type(changes)
        
        return UpdateContext(
            batch_id=f"{module}_{int(time.time())}",
            timestamp=datetime.now(),
            module=module,
            change_count=len(changes),
            file_types=file_types,
            update_type=update_type
        )
    
    def _classify_update_type(self, changes: List[FileChange]) -> str:
        """Classify the type of update based on changed files"""
        file_extensions = {Path(change.path).suffix.lower() for change in changes}
        file_names = {Path(change.path).name.lower() for change in changes}
        file_paths = {change.path.lower() for change in changes}
        
        # Test changes (check first as they may have code extensions but are still tests)
        if any('test' in name or 'spec' in name for name in file_names) or \
           any('test' in path or 'spec' in path for path in file_paths):
            return 'tests'
        
        # Code changes
        code_extensions = {'.py', '.js', '.ts', '.tsx', '.jsx', '.go', '.rs', '.java', '.cpp', '.c', '.h'}
        if any(ext in code_extensions for ext in file_extensions):
            return 'code'
        
        # Configuration changes
        config_extensions = {'.json', '.yaml', '.yml', '.toml', '.ini', '.cfg', '.conf'}
        if any(ext in config_extensions for ext in file_extensions):
            return 'config'
        
        # Documentation changes
        doc_extensions = {'.md', '.rst', '.txt'}
        if any(ext in doc_extensions for ext in file_extensions):
            return 'docs'
        
        return 'other'
    
    def _analyze_changes_for_knowledge(self, context: UpdateContext, changes: List[FileChange]) -> Optional[Dict[str, Any]]:
        """
        Analyze file changes to extract knowledge for the graph.
        
        This method looks for patterns, architectural decisions, and other
        knowledge that should be stored based on the file changes.
        """
        knowledge_items = []
        
        for change in changes:
            # Analyze individual file changes
            file_analysis = self._analyze_file_change(change, context)
            if file_analysis:
                knowledge_items.append(file_analysis)
        
        if not knowledge_items:
            return None
        
        # Create batch knowledge update
        return {
            'context': {
                'batch_id': context.batch_id,
                'timestamp': context.timestamp.isoformat(),
                'module': context.module,
                'update_type': context.update_type,
                'change_count': context.change_count
            },
            'items': knowledge_items
        }
    
    def _analyze_file_change(self, change: FileChange, context: UpdateContext) -> Optional[Dict[str, Any]]:
        """Analyze a single file change for knowledge extraction"""
        try:
            file_path = Path(change.path)
            
            # Basic file information
            analysis = {
                'file_path': change.path,
                'change_type': change.type,
                'priority': change.priority,
                'timestamp': change.timestamp,
                'file_extension': file_path.suffix,
                'file_size_kb': None
            }
            
            # Get file size if it exists
            try:
                if file_path.exists() and change.type != 'deleted':
                    analysis['file_size_kb'] = round(file_path.stat().st_size / 1024, 2)
            except:
                pass
            
            # Pattern detection based on file type and context
            patterns = self._detect_patterns(change, context)
            if patterns:
                analysis['patterns'] = patterns
                self.stats['patterns_detected'] += len(patterns)
            
            # Module relationship
            analysis['module'] = context.module
            analysis['update_context'] = context.update_type
            
            return analysis
            
        except Exception as e:
            self.logger.error(f"Failed to analyze file change {change.path}: {e}")
            return None
    
    def _detect_patterns(self, change: FileChange, context: UpdateContext) -> List[str]:
        """Detect patterns in file changes"""
        patterns = []
        file_path = Path(change.path)
        
        # Pattern: New Python module
        if change.type == 'created' and file_path.suffix == '.py':
            patterns.append('new_python_module')
        
        # Pattern: Configuration update
        if file_path.suffix in {'.json', '.yaml', '.yml'} and change.type == 'modified':
            patterns.append('configuration_change')
        
        # Pattern: Test file changes
        if 'test' in file_path.name.lower() or 'spec' in file_path.name.lower():
            patterns.append('test_change')
        
        # Pattern: Documentation updates
        if file_path.suffix in {'.md', '.rst'}:
            patterns.append('documentation_change')
        
        # Pattern: High priority changes (immediate attention)
        if change.priority == 0:  # IMMEDIATE priority
            patterns.append('high_priority_change')
        
        # Pattern: Batch changes (refactoring indicator)
        if context.change_count > 5:
            patterns.append('batch_refactoring')
        
        # Pattern: Cross-module changes
        if context.change_count > 1 and len(context.file_types) > 2:
            patterns.append('cross_cutting_change')
        
        return patterns
    
    def _store_knowledge_updates(self, context: UpdateContext, knowledge_updates: Dict[str, Any]) -> str:
        """Store knowledge updates in the knowledge system"""
        try:
            # Create a knowledge entry for this batch update
            content = {
                'title': f"Auto-update from {context.module} changes",
                'description': f"Automated knowledge update from {context.change_count} file changes",
                'context': knowledge_updates['context'],
                'items': knowledge_updates['items'],
                'detected_patterns': [
                    pattern for item in knowledge_updates['items'] 
                    for pattern in item.get('patterns', [])
                ],
                'automation_source': 'FileChangeDetector'
            }
            
            metadata = {
                'type': 'auto_update',
                'module': context.module,
                'update_type': context.update_type,
                'timestamp': context.timestamp.isoformat(),
                'change_count': context.change_count,
                'batch_id': context.batch_id
            }
            
            # Try to store in auto_updates collection first
            doc_id = None
            collection_to_use = 'auto_updates'
            
            try:
                doc_id = self.knowledge_system.store_knowledge(
                    collection_to_use,
                    content,
                    metadata,
                    context.batch_id
                )
            except Exception as auto_updates_error:
                self.logger.warning(f"auto_updates collection not available: {auto_updates_error}")
                
                # Fall back to learnings collection as alternative
                collection_to_use = 'learnings'
                self.logger.info(f"Falling back to {collection_to_use} collection")
                
                try:
                    # Adapt content for learnings collection
                    learning_content = {
                        'title': content['title'],
                        'description': content['description'] + ' (stored as learning due to auto_updates unavailability)',
                        'source': 'file_change_auto_update',
                        'issue_id': f"auto-update-{context.batch_id}",
                        'complexity': 'medium',
                        'success_factors': f"Detected {len(content['detected_patterns'])} patterns",
                        'recommendations': f"File changes in {context.module} module",
                        'raw_data': content
                    }
                    
                    learning_metadata = {
                        'type': 'learning',
                        'source': 'file_change_auto_update',
                        'module': context.module,
                        'timestamp': context.timestamp.isoformat(),
                        'batch_id': context.batch_id
                    }
                    
                    doc_id = self.knowledge_system.store_knowledge(
                        collection_to_use,
                        learning_content,
                        learning_metadata,
                        f"auto_update_{context.batch_id}"
                    )
                    
                except Exception as learnings_error:
                    self.logger.error(f"Failed to store in both auto_updates and learnings collections: {learnings_error}")
                    return None
            
            if doc_id:
                self.stats['updates_stored'] += 1
                self.logger.info(f"Stored knowledge update {doc_id} in {collection_to_use} for module {context.module}")
                return doc_id
            else:
                self.logger.error(f"Failed to store knowledge update for {context.module}")
                return None
                
        except Exception as e:
            self.logger.error(f"Error storing knowledge updates: {e}")
            return None
    
    def get_statistics(self) -> Dict[str, Any]:
        """Get updater statistics"""
        return {
            **self.stats,
            'knowledge_system_available': KNOWLEDGE_SYSTEM_AVAILABLE,
            'knowledge_path': str(self.knowledge_path)
        }
    
    def process_detector_batches(self, detector: FileChangeDetector) -> Dict[str, Any]:
        """
        Process all batched changes from a FileChangeDetector.
        
        Args:
            detector: FileChangeDetector instance with pending changes
            
        Returns:
            Processing results summary
        """
        batches = detector.batch_related_changes()
        
        results = {
            'processed_modules': [],
            'successful_updates': 0,
            'failed_updates': 0,
            'total_changes': 0
        }
        
        for module, changes in batches.items():
            self.logger.info(f"Processing {len(changes)} changes for module {module}")
            results['total_changes'] += len(changes)
            
            update_id = self.process_change_batch(module, changes)
            
            if update_id:
                results['successful_updates'] += 1
                results['processed_modules'].append({
                    'module': module,
                    'change_count': len(changes),
                    'update_id': update_id
                })
            else:
                results['failed_updates'] += 1
        
        return results


def create_auto_update_system(root_paths: Optional[List[str]] = None) -> tuple[FileChangeDetector, KnowledgeGraphUpdater]:
    """
    Create a complete auto-update system with file detection and knowledge updates.
    
    Args:
        root_paths: Paths to monitor for changes
        
    Returns:
        Tuple of (FileChangeDetector, KnowledgeGraphUpdater)
    """
    detector = FileChangeDetector(root_paths)
    updater = KnowledgeGraphUpdater()
    
    return detector, updater


def main():
    """Demo and testing for knowledge graph updater"""
    print("Knowledge Graph Auto-Update System Demo")
    print("======================================")
    
    # Create system components
    detector, updater = create_auto_update_system(["."])
    
    # Show initial status
    print(f"Detector status: {detector.get_status()}")
    print(f"Updater stats: {updater.get_statistics()}")
    
    # Simulate some changes
    test_files = [
        "src/core.py",
        "src/utils.py", 
        "config/settings.yaml",
        "tests/test_core.py"
    ]
    
    print(f"\nSimulating changes to {len(test_files)} files...")
    for test_file in test_files:
        detector.on_file_modified(test_file)
    
    # Process batches
    results = updater.process_detector_batches(detector)
    
    print(f"\nProcessing Results:")
    print(f"  Total changes: {results['total_changes']}")
    print(f"  Successful updates: {results['successful_updates']}")
    print(f"  Failed updates: {results['failed_updates']}")
    print(f"  Processed modules: {len(results['processed_modules'])}")
    
    for module_info in results['processed_modules']:
        print(f"    {module_info['module']}: {module_info['change_count']} changes -> {module_info['update_id']}")


if __name__ == "__main__":
    main()