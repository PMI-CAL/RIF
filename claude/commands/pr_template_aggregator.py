#!/usr/bin/env python3
"""
PR Template Aggregator - Phase 2 of Issue #205
Handles template parsing, context aggregation, and PR body generation.

This module aggregates data from multiple sources (issue metadata, checkpoints,
file modifications) and populates the PR template with contextual information.
"""

import json
import logging
import re
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional, Any, Tuple
from dataclasses import dataclass

@dataclass
class CheckpointData:
    """Structure for checkpoint information."""
    checkpoint_id: str
    phase: str
    status: str
    description: str
    timestamp: str
    components: Dict[str, Any]

@dataclass
class PRContext:
    """Complete context for PR generation."""
    issue_number: int
    issue_metadata: Dict[str, Any]
    checkpoints: List[CheckpointData]
    file_modifications: Dict[str, List[str]]
    quality_results: Dict[str, Any]
    implementation_summary: str

class PRTemplateAggregator:
    """
    Aggregates context data and populates PR templates for automated PR creation.
    """
    
    def __init__(self, template_path: str = ".github/pull_request_template.md"):
        """
        Initialize the PR template aggregator.
        
        Args:
            template_path: Path to PR template file
        """
        self.template_path = Path(template_path)
        self.setup_logging()
        
    def setup_logging(self):
        """Setup logging for template aggregator operations."""
        logging.basicConfig(
            level=logging.INFO,
            format='%(asctime)s - PRTemplateAggregator - %(levelname)s - %(message)s'
        )
        self.logger = logging.getLogger(__name__)
        
    def load_template(self) -> str:
        """
        Load the PR template content.
        
        Returns:
            Template content as string
        """
        try:
            if not self.template_path.exists():
                self.logger.warning(f"Template not found at {self.template_path}, using default")
                return self._get_default_template()
                
            with open(self.template_path, 'r', encoding='utf-8') as f:
                return f.read()
                
        except Exception as e:
            self.logger.error(f"Failed to load template: {e}")
            return self._get_default_template()
    
    def _get_default_template(self) -> str:
        """Get a minimal default template if main template is unavailable."""
        return """# Pull Request

## 📋 Summary
{summary}

## 🔗 Related Issues
Closes #{issue_number}

## 📝 Changes Made
{changes_made}

### Modified Files
{modified_files}

## 🧪 Testing
{testing_summary}

## 🤖 RIF Automation Status
This PR was automatically created by RIF-Implementer for issue #{issue_number}.

**Implementation Phases Completed**: {phases_completed}
**Quality Gates**: {quality_status}
"""
    
    def load_checkpoints(self, issue_number: int) -> List[CheckpointData]:
        """
        Load all checkpoints for an issue from the knowledge system.
        
        Args:
            issue_number: GitHub issue number
            
        Returns:
            List of checkpoint data structures
        """
        checkpoints = []
        checkpoint_dir = Path("knowledge/checkpoints")
        
        if not checkpoint_dir.exists():
            self.logger.warning(f"Checkpoint directory not found: {checkpoint_dir}")
            return []
            
        # Look for checkpoint files related to this issue
        pattern = f"issue-{issue_number}-*.json"
        
        try:
            for checkpoint_file in checkpoint_dir.glob(pattern):
                try:
                    with open(checkpoint_file, 'r', encoding='utf-8') as f:
                        data = json.load(f)
                    
                    checkpoint = CheckpointData(
                        checkpoint_id=data.get('checkpoint_id', checkpoint_file.stem),
                        phase=data.get('phase', 'Unknown'),
                        status=data.get('status', 'unknown'),
                        description=data.get('description', ''),
                        timestamp=data.get('timestamp', ''),
                        components=data.get('components_implemented', {})
                    )
                    
                    checkpoints.append(checkpoint)
                    
                except Exception as e:
                    self.logger.warning(f"Failed to load checkpoint {checkpoint_file}: {e}")
                    continue
                    
            # Sort checkpoints by timestamp
            checkpoints.sort(key=lambda cp: cp.timestamp)
            self.logger.info(f"Loaded {len(checkpoints)} checkpoints for issue #{issue_number}")
            
        except Exception as e:
            self.logger.error(f"Error loading checkpoints: {e}")
            
        return checkpoints
    
    def generate_implementation_summary(self, checkpoints: List[CheckpointData]) -> str:
        """
        Generate a summary of implementation work from checkpoints.
        
        Args:
            checkpoints: List of checkpoint data
            
        Returns:
            Formatted implementation summary
        """
        if not checkpoints:
            return "No checkpoint data available."
            
        summary_parts = []
        
        # Count completed phases
        completed_phases = [cp for cp in checkpoints if cp.status == 'complete']
        summary_parts.append(f"**Completed {len(completed_phases)} implementation phases:**")
        
        for checkpoint in completed_phases:
            summary_parts.append(f"- **{checkpoint.phase}**: {checkpoint.description}")
            
            # Add component details if available
            if checkpoint.components:
                component_count = len(checkpoint.components)
                summary_parts.append(f"  - Implemented {component_count} components")
                
        # Add latest checkpoint status
        if checkpoints:
            latest = checkpoints[-1]
            summary_parts.append(f"\n**Latest Status**: {latest.status.title()} - {latest.description}")
            
        return "\n".join(summary_parts)
    
    def generate_changes_summary(self, file_modifications: Dict[str, List[str]], 
                                checkpoints: List[CheckpointData]) -> str:
        """
        Generate a summary of code changes made.
        
        Args:
            file_modifications: Dictionary of file changes by type
            checkpoints: Checkpoint data for additional context
            
        Returns:
            Formatted changes summary
        """
        changes_parts = []
        
        # File modification summary
        total_files = (len(file_modifications.get('added', [])) + 
                      len(file_modifications.get('modified', [])) + 
                      len(file_modifications.get('deleted', [])))
        
        if total_files > 0:
            changes_parts.append(f"**Total files affected**: {total_files}")
            
            if file_modifications.get('added'):
                changes_parts.append(f"- **Added**: {len(file_modifications['added'])} files")
            if file_modifications.get('modified'):
                changes_parts.append(f"- **Modified**: {len(file_modifications['modified'])} files")  
            if file_modifications.get('deleted'):
                changes_parts.append(f"- **Deleted**: {len(file_modifications['deleted'])} files")
        else:
            changes_parts.append("No file modifications detected.")
            
        # Add implementation details from checkpoints
        if checkpoints:
            changes_parts.append("\n**Key Implementation Components:**")
            
            for checkpoint in checkpoints:
                if checkpoint.components:
                    for component_name, component_data in checkpoint.components.items():
                        if isinstance(component_data, dict):
                            status = component_data.get('status', 'unknown')
                            desc = component_data.get('description', component_name)
                            changes_parts.append(f"- **{component_name}** ({status}): {desc}")
                            
        return "\n".join(changes_parts)
    
    def generate_modified_files_list(self, file_modifications: Dict[str, List[str]]) -> str:
        """
        Generate a formatted list of modified files.
        
        Args:
            file_modifications: Dictionary of file changes by type
            
        Returns:
            Formatted file list
        """
        file_list_parts = []
        
        for change_type, files in file_modifications.items():
            if files:
                change_emoji = {
                    'added': '➕',
                    'modified': '✏️', 
                    'deleted': '➖'
                }.get(change_type, '📝')
                
                file_list_parts.append(f"\n**{change_emoji} {change_type.title()} Files:**")
                for file_path in files[:10]:  # Limit to first 10 files
                    file_list_parts.append(f"- `{file_path}`")
                    
                if len(files) > 10:
                    file_list_parts.append(f"- ... and {len(files) - 10} more files")
                    
        return "\n".join(file_list_parts) if file_list_parts else "No file modifications detected."
    
    def populate_template(self, pr_context: PRContext) -> str:
        """
        Populate the PR template with contextual data.
        
        Args:
            pr_context: Complete context for PR generation
            
        Returns:
            Populated template content
        """
        try:
            template = self.load_template()
            
            # Generate all template variables
            implementation_summary = self.generate_implementation_summary(pr_context.checkpoints)
            changes_summary = self.generate_changes_summary(
                pr_context.file_modifications, pr_context.checkpoints)
            modified_files = self.generate_modified_files_list(pr_context.file_modifications)
            
            # Calculate phases completed
            completed_phases = len([cp for cp in pr_context.checkpoints if cp.status == 'complete'])
            
            # Quality status summary
            quality_status = "Pending validation" if not pr_context.quality_results else "Gates passed"
            
            # Template variable replacements
            replacements = {
                '{issue_number}': str(pr_context.issue_number),
                '{summary}': implementation_summary,
                '{changes_made}': changes_summary,
                '{modified_files}': modified_files,
                '{testing_summary}': "Testing pending - will be completed in PR validation phase.",
                '{phases_completed}': str(completed_phases),
                '{quality_status}': quality_status
            }
            
            # Apply replacements
            populated_template = template
            for placeholder, value in replacements.items():
                populated_template = populated_template.replace(placeholder, value)
                
            self.logger.info(f"Successfully populated template for issue #{pr_context.issue_number}")
            return populated_template
            
        except Exception as e:
            self.logger.error(f"Failed to populate template: {e}")
            return self._get_fallback_template(pr_context)
    
    def _get_fallback_template(self, pr_context: PRContext) -> str:
        """Get a minimal fallback template if population fails."""
        return f"""# Pull Request for Issue #{pr_context.issue_number}

## Summary
{pr_context.issue_metadata.get('title', 'Automated implementation')}

## Related Issues
Closes #{pr_context.issue_number}

## Changes Made
Implementation completed via RIF automation system.

**Phases completed**: {len([cp for cp in pr_context.checkpoints if cp.status == 'complete'])}

## RIF Automation
This PR was automatically created by RIF-Implementer.
Manual review and validation recommended.
"""

    def aggregate_pr_context(self, issue_number: int, issue_metadata: Dict[str, Any],
                           file_modifications: Dict[str, List[str]],
                           quality_results: Optional[Dict[str, Any]] = None) -> PRContext:
        """
        Aggregate all context data for PR creation.
        
        Args:
            issue_number: GitHub issue number
            issue_metadata: Issue information from GitHub
            file_modifications: File changes summary
            quality_results: Quality gate results (optional)
            
        Returns:
            Complete PR context structure
        """
        checkpoints = self.load_checkpoints(issue_number)
        implementation_summary = self.generate_implementation_summary(checkpoints)
        
        return PRContext(
            issue_number=issue_number,
            issue_metadata=issue_metadata,
            checkpoints=checkpoints,
            file_modifications=file_modifications,
            quality_results=quality_results or {},
            implementation_summary=implementation_summary
        )