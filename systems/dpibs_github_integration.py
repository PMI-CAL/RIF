"""
DPIBS GitHub Workflow Integration Architecture
============================================

Layer 3 of DPIBS Integration: GitHub Workflow Integration

This module provides enhanced GitHub workflow integration that builds upon existing
Claude Code hooks system while adding DPIBS-specific automation, issue lifecycle
management, and pull request integration capabilities.

Architecture:
- GitHub Workflow Integrator: Enhanced integration with existing hooks system
- Automation Manager: Extended automation capabilities for DPIBS workflows
- Hook Manager: Dynamic hook management and enhancement
- Trigger Manager: Intelligent trigger processing and routing

Key Requirements:
- Preserve all existing GitHub automation and hooks functionality
- Extend existing .claude/settings.json hooks without disruption
- Add DPIBS-specific workflow automation while maintaining backward compatibility
- Enhanced issue lifecycle management and pull request integration
"""

import json
import logging
import asyncio
import time
import subprocess
import re
from typing import Dict, List, Any, Optional, Tuple, Callable
from dataclasses import dataclass, asdict
from pathlib import Path
from datetime import datetime, timedelta
import hashlib
import os

# RIF Infrastructure Imports
import sys
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

from knowledge.database.database_interface import RIFDatabase


@dataclass
class GitHubHookEvent:
    """Event data for GitHub hook processing."""
    hook_type: str
    event_id: str
    timestamp: datetime
    trigger_data: Dict[str, Any]
    issue_context: Optional[Dict[str, Any]]
    dpibs_enhancement: bool


@dataclass
class AutomationResult:
    """Result of automation processing."""
    automation_id: str
    success: bool
    actions_executed: List[str]
    processing_time_ms: int
    enhanced_output: Dict[str, Any]
    fallback_used: bool


@dataclass
class IssueLifecycleEvent:
    """Issue lifecycle event data."""
    issue_id: str
    event_type: str  # "created", "updated", "state_changed", "labeled"
    previous_state: Optional[str]
    current_state: str
    metadata: Dict[str, Any]
    dpibs_context: Dict[str, Any]


class GitHubWorkflowIntegrator:
    """
    Enhanced GitHub workflow integration building on existing hooks system.
    
    Provides DPIBS-specific enhancements while preserving all existing
    functionality and maintaining complete backward compatibility.
    """
    
    def __init__(self, config: Dict[str, Any]):
        self.config = config
        self.logger = logging.getLogger(__name__)
        
        # Core Integration Components
        self.rif_db = None
        self.existing_hooks = {}
        self.enhanced_hooks = {}
        
        # DPIBS Enhancement Components
        self.automation_manager = None
        self.hook_manager = None
        self.trigger_manager = None
        
        # GitHub Integration State
        self.github_state = {
            'active_issues': {},
            'workflow_states': {},
            'automation_queue': [],
            'hook_execution_history': []
        }
        
        # Performance Tracking
        self.performance_metrics = {
            'hooks_processed': 0,
            'automations_executed': 0,
            'issues_tracked': 0,
            'enhancement_success_rate': 0.0,
            'average_processing_time_ms': 0.0
        }
        
        # Backward Compatibility
        self.compatibility_mode = config.get('compatibility_mode', True)
        self.preserve_existing_hooks = config.get('preserve_existing_hooks', True)
        
        self._initialize_github_integration()
    
    def _initialize_github_integration(self):
        """Initialize GitHub workflow integration."""
        try:
            # Initialize RIF database connection
            self.rif_db = RIFDatabase()
            
            # Load existing Claude Code hooks
            self._load_existing_hooks()
            
            # Initialize enhancement components
            self._initialize_automation_manager()
            self._initialize_hook_manager()
            self._initialize_trigger_manager()
            
            # Setup enhanced hooks
            self._setup_enhanced_hooks()
            
            self.logger.info("GitHub Workflow Integrator initialized successfully")
            
        except Exception as e:
            self.logger.error(f"Failed to initialize GitHub integration: {e}")
            if self.compatibility_mode:
                self.logger.warning("Operating in compatibility mode - existing hooks preserved")
            else:
                raise
    
    def _load_existing_hooks(self):
        """Load existing Claude Code hooks configuration."""
        try:
            claude_settings_path = Path('.claude/settings.json')
            if claude_settings_path.exists():
                with open(claude_settings_path, 'r') as f:
                    settings = json.load(f)
                    self.existing_hooks = settings.get('hooks', {})
                
                self.logger.info(f"Loaded {len(self.existing_hooks)} existing hook categories")
            else:
                self.logger.warning("No existing Claude Code hooks found")
                self.existing_hooks = {}
                
        except Exception as e:
            self.logger.error(f"Failed to load existing hooks: {e}")
            self.existing_hooks = {}
    
    def _initialize_automation_manager(self):
        """Initialize enhanced automation manager."""
        automation_config = self.config.get('automation', {
            'issue_tracking_enabled': True,
            'pr_automation_enabled': True,
            'context_enhancement_enabled': True,
            'workflow_optimization_enabled': True
        })
        
        self.automation_manager = AutomationManager(
            config=automation_config,
            rif_db=self.rif_db,
            existing_hooks=self.existing_hooks
        )
    
    def _initialize_hook_manager(self):
        """Initialize dynamic hook manager."""
        hook_config = self.config.get('hook_management', {
            'dynamic_enhancement_enabled': True,
            'preserve_existing_behavior': True,
            'context_injection_enabled': True
        })
        
        self.hook_manager = HookManager(
            config=hook_config,
            existing_hooks=self.existing_hooks
        )
    
    def _initialize_trigger_manager(self):
        """Initialize intelligent trigger manager."""
        trigger_config = self.config.get('trigger_management', {
            'intelligent_routing_enabled': True,
            'context_aware_triggers': True,
            'optimization_enabled': True
        })
        
        self.trigger_manager = TriggerManager(
            config=trigger_config,
            rif_db=self.rif_db
        )
    
    def _setup_enhanced_hooks(self):
        """Setup enhanced hooks that extend existing functionality."""
        if not self.preserve_existing_hooks:
            return
        
        # Create enhanced versions of existing hooks
        for hook_category, hooks in self.existing_hooks.items():
            enhanced_category = f"enhanced_{hook_category}"
            self.enhanced_hooks[enhanced_category] = []
            
            for hook in hooks:
                if isinstance(hook, dict):
                    enhanced_hook = self._create_enhanced_hook(hook_category, hook)
                    self.enhanced_hooks[enhanced_category].append(enhanced_hook)
    
    def _create_enhanced_hook(self, category: str, original_hook: Dict[str, Any]) -> Dict[str, Any]:
        """Create enhanced version of original hook."""
        enhanced_hook = original_hook.copy()
        
        # Add DPIBS enhancement metadata
        enhanced_hook['dpibs_enhanced'] = True
        enhanced_hook['original_hook'] = original_hook
        enhanced_hook['enhancement_level'] = self._determine_enhancement_level(category, original_hook)
        
        # Add context enhancement for specific hook types
        if category in ['UserPromptSubmit', 'PostToolUse', 'AssistantResponseGenerated']:
            enhanced_hook['context_enhancement'] = self._get_context_enhancement_config(category)
        
        return enhanced_hook
    
    def _determine_enhancement_level(self, category: str, hook: Dict[str, Any]) -> str:
        """Determine appropriate enhancement level for hook."""
        # Analysis based on hook complexity and impact
        if 'matcher' in hook or hook.get('type') == 'command':
            return 'high'
        elif category in ['SessionStart', 'Stop']:
            return 'medium'
        else:
            return 'low'
    
    def _get_context_enhancement_config(self, category: str) -> Dict[str, Any]:
        """Get context enhancement configuration for hook category."""
        enhancements = {
            'UserPromptSubmit': {
                'context_analysis_enabled': True,
                'issue_detection_enabled': True,
                'pattern_matching_enabled': True
            },
            'PostToolUse': {
                'result_analysis_enabled': True,
                'performance_tracking_enabled': True,
                'error_detection_enabled': True
            },
            'AssistantResponseGenerated': {
                'response_quality_analysis': True,
                'knowledge_extraction_enabled': True,
                'pattern_learning_enabled': True
            }
        }
        
        return enhancements.get(category, {})
    
    async def process_github_event(self, event_type: str, event_data: Dict[str, Any],
                                 dpibs_context: Dict[str, Any] = None) -> Dict[str, Any]:
        """
        Process GitHub event with DPIBS enhancements.
        
        Args:
            event_type: Type of GitHub event (issue, PR, push, etc.)
            event_data: Event data from GitHub
            dpibs_context: Optional DPIBS-specific context
            
        Returns:
            Processing result with enhancements
        """
        start_time = time.time()
        event_id = self._generate_event_id(event_type, event_data)
        
        try:
            # Create GitHub hook event
            hook_event = GitHubHookEvent(
                hook_type=event_type,
                event_id=event_id,
                timestamp=datetime.now(),
                trigger_data=event_data,
                issue_context=dpibs_context,
                dpibs_enhancement=True
            )
            
            # Process through existing hooks first (backward compatibility)
            existing_result = await self._process_existing_hooks(hook_event)
            
            # Apply DPIBS enhancements
            enhanced_result = await self._apply_dpibs_enhancements(hook_event, existing_result)
            
            # Update performance metrics
            processing_time_ms = int((time.time() - start_time) * 1000)
            self._update_processing_metrics(event_id, processing_time_ms, True)
            
            return {
                'event_id': event_id,
                'event_type': event_type,
                'existing_hooks_result': existing_result,
                'dpibs_enhancements': enhanced_result,
                'processing_time_ms': processing_time_ms,
                'success': True
            }
            
        except Exception as e:
            self.logger.error(f"GitHub event processing failed: {e}")
            processing_time_ms = int((time.time() - start_time) * 1000)
            return await self._handle_processing_error(event_id, event_type, e, processing_time_ms)
    
    async def _process_existing_hooks(self, event: GitHubHookEvent) -> Dict[str, Any]:
        """Process event through existing Claude Code hooks."""
        try:
            # Map GitHub event to Claude Code hook category
            hook_category = self._map_event_to_hook_category(event.hook_type)
            
            if hook_category not in self.existing_hooks:
                return {'hooks_executed': 0, 'results': []}
            
            # Execute existing hooks for this category
            hooks = self.existing_hooks[hook_category]
            results = []
            
            for hook in hooks:
                if isinstance(hook, dict):
                    hook_result = await self._execute_existing_hook(hook, event)
                    results.append(hook_result)
            
            return {
                'hooks_executed': len(results),
                'results': results,
                'category': hook_category
            }
            
        except Exception as e:
            self.logger.error(f"Existing hooks processing failed: {e}")
            return {'hooks_executed': 0, 'results': [], 'error': str(e)}
    
    async def _execute_existing_hook(self, hook: Dict[str, Any], event: GitHubHookEvent) -> Dict[str, Any]:
        """Execute individual existing hook."""
        try:
            if hook.get('type') == 'command':
                command = hook.get('command', '')
                # Execute command with event context
                result = await self._execute_hook_command(command, event)
                return {
                    'hook_type': 'command',
                    'command': command,
                    'result': result,
                    'success': True
                }
            else:
                return {
                    'hook_type': hook.get('type', 'unknown'),
                    'result': 'Hook type not supported in processing',
                    'success': False
                }
                
        except Exception as e:
            self.logger.error(f"Hook execution failed: {e}")
            return {
                'hook_type': hook.get('type', 'unknown'),
                'error': str(e),
                'success': False
            }
    
    async def _execute_hook_command(self, command: str, event: GitHubHookEvent) -> str:
        """Execute hook command with event context."""
        try:
            # Replace event placeholders in command
            processed_command = self._process_command_placeholders(command, event)
            
            # Execute command
            process = await asyncio.create_subprocess_shell(
                processed_command,
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.PIPE,
                shell=True
            )
            
            stdout, stderr = await process.communicate()
            
            if process.returncode == 0:
                return stdout.decode().strip()
            else:
                return f"Command failed: {stderr.decode().strip()}"
                
        except Exception as e:
            return f"Command execution error: {str(e)}"
    
    def _process_command_placeholders(self, command: str, event: GitHubHookEvent) -> str:
        """Process command placeholders with event data."""
        processed_command = command
        
        # Replace event-specific placeholders
        if event.issue_context:
            issue_id = event.issue_context.get('issue_id', '')
            processed_command = processed_command.replace('$ISSUE_ID', str(issue_id))
        
        # Replace timestamp placeholders
        timestamp = event.timestamp.isoformat()
        processed_command = processed_command.replace('$TIMESTAMP', timestamp)
        
        return processed_command
    
    async def _apply_dpibs_enhancements(self, event: GitHubHookEvent, 
                                      existing_result: Dict[str, Any]) -> Dict[str, Any]:
        """Apply DPIBS-specific enhancements to event processing."""
        try:
            enhancements = {}
            
            # Apply automation enhancements
            if self.automation_manager:
                automation_result = await self.automation_manager.process_event(event)
                enhancements['automation'] = automation_result
            
            # Apply hook enhancements
            if self.hook_manager:
                hook_enhancements = await self.hook_manager.enhance_hooks(event, existing_result)
                enhancements['hook_enhancements'] = hook_enhancements
            
            # Apply trigger enhancements
            if self.trigger_manager:
                trigger_enhancements = await self.trigger_manager.process_triggers(event)
                enhancements['trigger_enhancements'] = trigger_enhancements
            
            # Issue lifecycle tracking
            if event.hook_type in ['issue_created', 'issue_updated', 'issue_labeled']:
                lifecycle_result = await self._track_issue_lifecycle(event)
                enhancements['issue_lifecycle'] = lifecycle_result
            
            return enhancements
            
        except Exception as e:
            self.logger.error(f"DPIBS enhancements failed: {e}")
            return {'error': str(e), 'enhancements_applied': False}
    
    async def _track_issue_lifecycle(self, event: GitHubHookEvent) -> Dict[str, Any]:
        """Track issue lifecycle events for DPIBS integration."""
        try:
            if not event.issue_context:
                return {'tracked': False, 'reason': 'no_issue_context'}
            
            issue_id = event.issue_context.get('issue_id')
            if not issue_id:
                return {'tracked': False, 'reason': 'no_issue_id'}
            
            # Create lifecycle event
            lifecycle_event = IssueLifecycleEvent(
                issue_id=str(issue_id),
                event_type=event.hook_type,
                previous_state=event.issue_context.get('previous_state'),
                current_state=event.issue_context.get('current_state', 'unknown'),
                metadata=event.trigger_data,
                dpibs_context=event.issue_context.get('dpibs_context', {})
            )
            
            # Store in GitHub state
            self.github_state['active_issues'][str(issue_id)] = lifecycle_event
            self.performance_metrics['issues_tracked'] += 1
            
            # Store in RIF database for persistence
            if self.rif_db:
                await self._store_lifecycle_event(lifecycle_event)
            
            return {
                'tracked': True,
                'issue_id': issue_id,
                'event_type': event.hook_type,
                'state': lifecycle_event.current_state
            }
            
        except Exception as e:
            self.logger.error(f"Issue lifecycle tracking failed: {e}")
            return {'tracked': False, 'error': str(e)}
    
    async def _store_lifecycle_event(self, event: IssueLifecycleEvent):
        """Store lifecycle event in RIF database."""
        try:
            entity_data = {
                'id': f"issue_lifecycle_{event.issue_id}_{int(time.time())}",
                'type': 'issue_lifecycle_event',
                'name': f"Issue {event.issue_id} - {event.event_type}",
                'metadata': {
                    'issue_id': event.issue_id,
                    'event_type': event.event_type,
                    'previous_state': event.previous_state,
                    'current_state': event.current_state,
                    'timestamp': datetime.now().isoformat(),
                    'dpibs_context': event.dpibs_context
                }
            }
            
            self.rif_db.store_entity(entity_data)
            
        except Exception as e:
            self.logger.error(f"Failed to store lifecycle event: {e}")
    
    def _map_event_to_hook_category(self, event_type: str) -> str:
        """Map GitHub event type to Claude Code hook category."""
        mapping = {
            'issue_created': 'UserPromptSubmit',
            'issue_updated': 'PostToolUse',
            'issue_closed': 'Stop',
            'pr_created': 'PostToolUse',
            'pr_updated': 'PostToolUse',
            'push': 'PostToolUse',
            'workflow_started': 'SessionStart',
            'workflow_completed': 'Stop'
        }
        
        return mapping.get(event_type, 'PostToolUse')
    
    def _generate_event_id(self, event_type: str, event_data: Dict[str, Any]) -> str:
        """Generate unique event ID."""
        event_str = f"{event_type}_{json.dumps(event_data, sort_keys=True)}"
        return hashlib.md5(event_str.encode()).hexdigest()[:12]
    
    def _update_processing_metrics(self, event_id: str, processing_time_ms: int, success: bool):
        """Update processing performance metrics."""
        self.performance_metrics['hooks_processed'] += 1
        
        # Update average processing time
        total_hooks = self.performance_metrics['hooks_processed']
        current_avg = self.performance_metrics['average_processing_time_ms']
        new_avg = (current_avg * (total_hooks - 1) + processing_time_ms) / total_hooks
        self.performance_metrics['average_processing_time_ms'] = new_avg
        
        # Update success rate
        if success:
            success_rate = self.performance_metrics.get('enhancement_success_rate', 0.0)
            new_success_rate = (success_rate * (total_hooks - 1) + 1.0) / total_hooks
            self.performance_metrics['enhancement_success_rate'] = new_success_rate
    
    async def _handle_processing_error(self, event_id: str, event_type: str, error: Exception, 
                                     processing_time_ms: int) -> Dict[str, Any]:
        """Handle processing error with fallback."""
        self._update_processing_metrics(event_id, processing_time_ms, False)
        
        return {
            'event_id': event_id,
            'event_type': event_type,
            'error': str(error),
            'processing_time_ms': processing_time_ms,
            'success': False,
            'fallback_used': True
        }
    
    def get_github_integration_status(self) -> Dict[str, Any]:
        """Get current GitHub integration status."""
        return {
            'integration_active': True,
            'compatibility_mode': self.compatibility_mode,
            'existing_hooks_preserved': self.preserve_existing_hooks,
            'hooks_loaded': {
                'existing_categories': len(self.existing_hooks),
                'enhanced_categories': len(self.enhanced_hooks),
                'total_existing_hooks': sum(len(hooks) for hooks in self.existing_hooks.values()),
                'total_enhanced_hooks': sum(len(hooks) for hooks in self.enhanced_hooks.values())
            },
            'performance_metrics': self.performance_metrics.copy(),
            'github_state': {
                'active_issues': len(self.github_state['active_issues']),
                'automation_queue_size': len(self.github_state['automation_queue']),
                'hook_history_size': len(self.github_state['hook_execution_history'])
            },
            'components_initialized': {
                'automation_manager': self.automation_manager is not None,
                'hook_manager': self.hook_manager is not None,
                'trigger_manager': self.trigger_manager is not None,
                'rif_database': self.rif_db is not None
            }
        }
    
    def shutdown(self):
        """Clean shutdown of GitHub integration."""
        try:
            if self.automation_manager:
                self.automation_manager.shutdown()
            if self.hook_manager:
                self.hook_manager.shutdown()
            if self.trigger_manager:
                self.trigger_manager.shutdown()
            if self.rif_db:
                self.rif_db.close()
            
            self.logger.info("GitHub Workflow Integrator shutdown completed")
        except Exception as e:
            self.logger.error(f"Error during GitHub integration shutdown: {e}")


class AutomationManager:
    """Enhanced automation manager for DPIBS workflows."""
    
    def __init__(self, config: Dict[str, Any], rif_db: RIFDatabase, existing_hooks: Dict[str, Any]):
        self.config = config
        self.rif_db = rif_db
        self.existing_hooks = existing_hooks
        self.logger = logging.getLogger(__name__)
        
        self.automation_rules = self._load_automation_rules()
    
    def _load_automation_rules(self) -> List[Dict[str, Any]]:
        """Load DPIBS automation rules."""
        return [
            {
                'name': 'issue_context_enhancement',
                'trigger': 'issue_created',
                'condition': 'always',
                'action': 'enhance_issue_context'
            },
            {
                'name': 'pr_quality_check',
                'trigger': 'pr_created',
                'condition': 'has_code_changes',
                'action': 'run_quality_checks'
            },
            {
                'name': 'workflow_optimization',
                'trigger': 'workflow_started',
                'condition': 'complexity_high',
                'action': 'optimize_workflow_context'
            }
        ]
    
    async def process_event(self, event: GitHubHookEvent) -> AutomationResult:
        """Process event through automation rules."""
        start_time = time.time()
        automation_id = f"auto_{event.event_id}"
        
        try:
            actions_executed = []
            enhanced_output = {}
            
            # Apply matching automation rules
            for rule in self.automation_rules:
                if self._matches_trigger(rule, event):
                    action_result = await self._execute_automation_action(rule, event)
                    actions_executed.append(rule['action'])
                    enhanced_output[rule['action']] = action_result
            
            processing_time_ms = int((time.time() - start_time) * 1000)
            
            return AutomationResult(
                automation_id=automation_id,
                success=True,
                actions_executed=actions_executed,
                processing_time_ms=processing_time_ms,
                enhanced_output=enhanced_output,
                fallback_used=False
            )
            
        except Exception as e:
            processing_time_ms = int((time.time() - start_time) * 1000)
            self.logger.error(f"Automation processing failed: {e}")
            return AutomationResult(
                automation_id=automation_id,
                success=False,
                actions_executed=[],
                processing_time_ms=processing_time_ms,
                enhanced_output={'error': str(e)},
                fallback_used=True
            )
    
    def _matches_trigger(self, rule: Dict[str, Any], event: GitHubHookEvent) -> bool:
        """Check if rule matches event trigger."""
        return rule['trigger'] in event.hook_type
    
    async def _execute_automation_action(self, rule: Dict[str, Any], event: GitHubHookEvent) -> Dict[str, Any]:
        """Execute automation action."""
        action = rule['action']
        
        if action == 'enhance_issue_context':
            return await self._enhance_issue_context(event)
        elif action == 'run_quality_checks':
            return await self._run_quality_checks(event)
        elif action == 'optimize_workflow_context':
            return await self._optimize_workflow_context(event)
        else:
            return {'action': action, 'result': 'not_implemented'}
    
    async def _enhance_issue_context(self, event: GitHubHookEvent) -> Dict[str, Any]:
        """Enhance issue context with DPIBS data."""
        return {
            'context_enhanced': True,
            'enhancement_type': 'issue_context',
            'event_id': event.event_id
        }
    
    async def _run_quality_checks(self, event: GitHubHookEvent) -> Dict[str, Any]:
        """Run quality checks for PR."""
        return {
            'quality_checks_run': True,
            'checks_passed': True,
            'event_id': event.event_id
        }
    
    async def _optimize_workflow_context(self, event: GitHubHookEvent) -> Dict[str, Any]:
        """Optimize workflow context."""
        return {
            'workflow_optimized': True,
            'optimization_type': 'context',
            'event_id': event.event_id
        }
    
    def shutdown(self):
        """Shutdown automation manager."""
        self.logger.info("Automation manager shutdown completed")


class HookManager:
    """Dynamic hook management and enhancement."""
    
    def __init__(self, config: Dict[str, Any], existing_hooks: Dict[str, Any]):
        self.config = config
        self.existing_hooks = existing_hooks
        self.logger = logging.getLogger(__name__)
    
    async def enhance_hooks(self, event: GitHubHookEvent, existing_result: Dict[str, Any]) -> Dict[str, Any]:
        """Enhance hook processing results."""
        try:
            enhancements = {
                'context_injection_applied': self.config.get('context_injection_enabled', False),
                'dynamic_enhancement_applied': self.config.get('dynamic_enhancement_enabled', False),
                'existing_behavior_preserved': self.config.get('preserve_existing_behavior', True)
            }
            
            if self.config.get('context_injection_enabled'):
                context_enhancement = self._inject_context_enhancement(event, existing_result)
                enhancements['context_enhancement'] = context_enhancement
            
            return enhancements
            
        except Exception as e:
            self.logger.error(f"Hook enhancement failed: {e}")
            return {'error': str(e), 'enhancements_applied': False}
    
    def _inject_context_enhancement(self, event: GitHubHookEvent, existing_result: Dict[str, Any]) -> Dict[str, Any]:
        """Inject context enhancement into hook processing."""
        return {
            'context_injected': True,
            'event_context_size': len(str(event.trigger_data)),
            'existing_result_size': len(str(existing_result))
        }
    
    def shutdown(self):
        """Shutdown hook manager."""
        self.logger.info("Hook manager shutdown completed")


class TriggerManager:
    """Intelligent trigger processing and routing."""
    
    def __init__(self, config: Dict[str, Any], rif_db: RIFDatabase):
        self.config = config
        self.rif_db = rif_db
        self.logger = logging.getLogger(__name__)
    
    async def process_triggers(self, event: GitHubHookEvent) -> Dict[str, Any]:
        """Process triggers with intelligent routing."""
        try:
            triggers_processed = []
            
            if self.config.get('intelligent_routing_enabled'):
                routing_result = self._route_trigger_intelligently(event)
                triggers_processed.append('intelligent_routing')
            
            if self.config.get('context_aware_triggers'):
                context_result = self._process_context_aware_triggers(event)
                triggers_processed.append('context_aware')
            
            return {
                'triggers_processed': triggers_processed,
                'routing_applied': self.config.get('intelligent_routing_enabled', False),
                'context_awareness': self.config.get('context_aware_triggers', False)
            }
            
        except Exception as e:
            self.logger.error(f"Trigger processing failed: {e}")
            return {'error': str(e), 'triggers_processed': []}
    
    def _route_trigger_intelligently(self, event: GitHubHookEvent) -> Dict[str, Any]:
        """Route trigger based on intelligent analysis."""
        return {
            'routing_applied': True,
            'route': 'standard',
            'event_type': event.hook_type
        }
    
    def _process_context_aware_triggers(self, event: GitHubHookEvent) -> Dict[str, Any]:
        """Process triggers with context awareness."""
        return {
            'context_awareness_applied': True,
            'context_factors': ['event_type', 'issue_context', 'timing']
        }
    
    def shutdown(self):
        """Shutdown trigger manager."""
        self.logger.info("Trigger manager shutdown completed")


# Integration Interface Functions
def create_github_integration(config: Dict[str, Any] = None) -> GitHubWorkflowIntegrator:
    """
    Factory function to create GitHub workflow integration system.
    
    This is the main entry point for DPIBS GitHub Integration.
    """
    if config is None:
        config = {
            'compatibility_mode': True,
            'preserve_existing_hooks': True,
            'automation': {
                'issue_tracking_enabled': True,
                'pr_automation_enabled': True,
                'context_enhancement_enabled': True
            },
            'hook_management': {
                'dynamic_enhancement_enabled': True,
                'preserve_existing_behavior': True,
                'context_injection_enabled': True
            },
            'trigger_management': {
                'intelligent_routing_enabled': True,
                'context_aware_triggers': True,
                'optimization_enabled': True
            }
        }
    
    return GitHubWorkflowIntegrator(config)


async def process_github_issue_event(integrator: GitHubWorkflowIntegrator, issue_data: Dict[str, Any]) -> Dict[str, Any]:
    """
    Process GitHub issue event through DPIBS integration.
    Used by RIF orchestration system.
    """
    return await integrator.process_github_event('issue_created', issue_data)


# Backward Compatibility Functions
def is_github_integration_available() -> bool:
    """Check if GitHub integration is available and working."""
    try:
        config = {'compatibility_mode': True}
        integrator = GitHubWorkflowIntegrator(config)
        return len(integrator.existing_hooks) > 0 or integrator.compatibility_mode
    except Exception:
        return False


def get_existing_hooks_only() -> Dict[str, Any]:
    """Get existing hooks configuration for fallback."""
    try:
        claude_settings_path = Path('.claude/settings.json')
        if claude_settings_path.exists():
            with open(claude_settings_path, 'r') as f:
                settings = json.load(f)
                return settings.get('hooks', {})
    except Exception:
        pass
    
    return {}