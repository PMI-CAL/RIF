#!/usr/bin/env python3
"""
Configuration Manager for Adaptive Threshold Learning System
Issue #95: Adaptive Threshold Learning System

Manages safe configuration updates with rollback capabilities for adaptive threshold system.
"""

import json
import logging
import shutil
import yaml
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Dict, List, Optional, Any, Tuple
from dataclasses import dataclass, asdict

@dataclass
class ConfigurationCheckpoint:
    """Represents a configuration checkpoint for rollback."""
    checkpoint_id: str
    timestamp: str
    description: str
    configuration: Dict[str, Any]
    file_path: str
    backup_path: str

@dataclass
class ConfigurationChange:
    """Represents a configuration change."""
    change_id: str
    timestamp: str
    component_type: str
    field: str
    old_value: Any
    new_value: Any
    reason: str
    approved_by: Optional[str] = None

class ConfigurationManager:
    """
    Manages configuration changes with safety features and rollback capabilities.
    
    Features:
    - Safe configuration updates with validation
    - Checkpoint creation for rollback
    - Change tracking and audit trail
    - Backup management
    - Configuration validation
    """
    
    def __init__(self, config_dir: str = "config"):
        """
        Initialize the configuration manager.
        
        Args:
            config_dir: Directory containing configuration files
        """
        self.config_dir = Path(config_dir)
        self.config_dir.mkdir(parents=True, exist_ok=True)
        
        self.backup_dir = self.config_dir / "backups"
        self.backup_dir.mkdir(parents=True, exist_ok=True)
        
        self.checkpoints_file = self.config_dir / "checkpoints.jsonl"
        self.changes_file = self.config_dir / "changes.jsonl"
        
        self.setup_logging()
        self._initialize_config_files()
        
    def setup_logging(self):
        """Setup logging for configuration manager."""
        logging.basicConfig(
            level=logging.INFO,
            format='%(asctime)s - ConfigurationManager - %(levelname)s - %(message)s'
        )
        self.logger = logging.getLogger(__name__)
    
    def _initialize_config_files(self):
        """Initialize configuration files if they don't exist."""
        # Main threshold configuration
        threshold_config_file = self.config_dir / "rif-workflow.yaml"
        if not threshold_config_file.exists():
            self._create_default_threshold_config()
        
        # Adaptive configuration
        adaptive_config_file = self.config_dir / "adaptive-thresholds.yaml"
        if not adaptive_config_file.exists():
            self._create_default_adaptive_config()
        
        # Checkpoint and change tracking files
        for file_path in [self.checkpoints_file, self.changes_file]:
            if not file_path.exists():
                file_path.touch()
    
    def _create_default_threshold_config(self):
        """Create default threshold configuration."""
        default_config = {
            "quality_gates": {
                "coverage": {
                    "enabled": True,
                    "thresholds": {
                        "critical_algorithms": 95.0,
                        "public_apis": 90.0,
                        "business_logic": 85.0,
                        "integration_code": 80.0,
                        "ui_components": 75.0,
                        "test_utilities": 70.0
                    }
                },
                "security": {
                    "enabled": True,
                    "threshold": 90.0
                },
                "performance": {
                    "enabled": True,
                    "threshold": 85.0
                }
            },
            "adaptive_thresholds": {
                "enabled": True,
                "optimization_frequency": "weekly",
                "min_confidence": 0.7
            }
        }
        
        try:
            with open(self.config_dir / "rif-workflow.yaml", 'w') as f:
                yaml.dump(default_config, f, default_flow_style=False, indent=2)
            
            self.logger.info("Created default threshold configuration")
            
        except Exception as e:
            self.logger.error(f"Failed to create default threshold config: {e}")
    
    def _create_default_adaptive_config(self):
        """Create default adaptive configuration."""
        default_config = {
            "optimization": {
                "enabled": True,
                "frequency_days": 14,
                "min_confidence_threshold": 0.7,
                "max_threshold_change_percent": 20.0,
                "min_historical_data_days": 30,
                "required_sample_size": 10
            },
            "safety": {
                "require_manual_approval": True,
                "max_simultaneous_changes": 3,
                "rollback_monitoring_hours": 24,
                "performance_degradation_threshold": 0.1
            },
            "notifications": {
                "threshold_changes": True,
                "optimization_results": True,
                "rollback_required": True
            }
        }
        
        try:
            with open(self.config_dir / "adaptive-thresholds.yaml", 'w') as f:
                yaml.dump(default_config, f, default_flow_style=False, indent=2)
            
            self.logger.info("Created default adaptive configuration")
            
        except Exception as e:
            self.logger.error(f"Failed to create default adaptive config: {e}")
    
    def get_current_thresholds(self) -> Dict[str, float]:
        """
        Get current threshold configuration.
        
        Returns:
            Dictionary of component types to thresholds
        """
        try:
            config_file = self.config_dir / "rif-workflow.yaml"
            
            if not config_file.exists():
                self.logger.warning("Configuration file not found, using defaults")
                return self._get_default_thresholds()
            
            with open(config_file, 'r') as f:
                config = yaml.safe_load(f)
            
            thresholds = {}
            coverage_thresholds = config.get("quality_gates", {}).get("coverage", {}).get("thresholds", {})
            
            for component_type, threshold in coverage_thresholds.items():
                thresholds[component_type] = float(threshold)
            
            return thresholds
            
        except Exception as e:
            self.logger.error(f"Failed to get current thresholds: {e}")
            return self._get_default_thresholds()
    
    def _get_default_thresholds(self) -> Dict[str, float]:
        """Get default threshold values."""
        return {
            "critical_algorithms": 95.0,
            "public_apis": 90.0,
            "business_logic": 85.0,
            "integration_code": 80.0,
            "ui_components": 75.0,
            "test_utilities": 70.0
        }
    
    def create_checkpoint(self, 
                         checkpoint_id: str,
                         configuration: Optional[Dict[str, Any]] = None,
                         description: str = "") -> bool:
        """
        Create a configuration checkpoint for rollback.
        
        Args:
            checkpoint_id: Unique identifier for checkpoint
            configuration: Specific configuration to checkpoint (None for all)
            description: Description of the checkpoint
            
        Returns:
            True if checkpoint created successfully
        """
        try:
            timestamp = datetime.utcnow().isoformat() + "Z"
            
            if configuration is None:
                # Checkpoint all configurations
                configuration = self._get_all_configurations()
            
            # Create backup files
            backup_paths = self._create_backup_files(checkpoint_id)
            
            # Create checkpoint record
            checkpoint = ConfigurationCheckpoint(
                checkpoint_id=checkpoint_id,
                timestamp=timestamp,
                description=description,
                configuration=configuration,
                file_path=str(self.config_dir),
                backup_path=str(self.backup_dir / checkpoint_id)
            )
            
            # Save checkpoint record
            with open(self.checkpoints_file, 'a') as f:
                f.write(json.dumps(asdict(checkpoint)) + "\n")
            
            self.logger.info(f"Created checkpoint: {checkpoint_id}")
            return True
            
        except Exception as e:
            self.logger.error(f"Failed to create checkpoint {checkpoint_id}: {e}")
            return False
    
    def _get_all_configurations(self) -> Dict[str, Any]:
        """Get all current configurations."""
        configurations = {}
        
        config_files = [
            "rif-workflow.yaml",
            "adaptive-thresholds.yaml",
            "component-types.yaml"
        ]
        
        for config_file in config_files:
            file_path = self.config_dir / config_file
            if file_path.exists():
                try:
                    with open(file_path, 'r') as f:
                        if config_file.endswith('.yaml'):
                            configurations[config_file] = yaml.safe_load(f)
                        else:
                            configurations[config_file] = json.load(f)
                except Exception as e:
                    self.logger.warning(f"Failed to load {config_file}: {e}")
        
        return configurations
    
    def _create_backup_files(self, checkpoint_id: str) -> List[str]:
        """Create backup files for checkpoint."""
        backup_paths = []
        checkpoint_backup_dir = self.backup_dir / checkpoint_id
        checkpoint_backup_dir.mkdir(parents=True, exist_ok=True)
        
        # Copy all configuration files to backup directory
        for config_file in self.config_dir.glob("*.yaml"):
            if config_file.is_file():
                backup_path = checkpoint_backup_dir / config_file.name
                shutil.copy2(config_file, backup_path)
                backup_paths.append(str(backup_path))
        
        return backup_paths
    
    def update_threshold(self, 
                        component_type: str,
                        new_threshold: float,
                        reason: str,
                        approved_by: Optional[str] = None,
                        create_checkpoint_before: bool = True) -> bool:
        """
        Update a threshold with change tracking.
        
        Args:
            component_type: Component type to update
            new_threshold: New threshold value
            reason: Reason for the change
            approved_by: Who approved the change
            create_checkpoint_before: Whether to create checkpoint before change
            
        Returns:
            True if update was successful
        """
        try:
            # Create checkpoint if requested
            if create_checkpoint_before:
                checkpoint_id = f"pre_update_{component_type}_{int(datetime.utcnow().timestamp())}"
                self.create_checkpoint(checkpoint_id, description=f"Pre-update checkpoint for {component_type}")
            
            # Get current configuration
            current_thresholds = self.get_current_thresholds()
            old_threshold = current_thresholds.get(component_type, 80.0)
            
            # Validate new threshold
            if not self._validate_threshold_value(component_type, new_threshold):
                self.logger.error(f"Invalid threshold value: {new_threshold} for {component_type}")
                return False
            
            # Update configuration
            success = self._update_configuration_file(component_type, new_threshold)
            
            if success:
                # Record the change
                self._record_configuration_change(
                    component_type, "threshold", old_threshold, new_threshold, reason, approved_by
                )
                
                self.logger.info(f"Updated threshold for {component_type}: {old_threshold} -> {new_threshold}")
                return True
            else:
                return False
                
        except Exception as e:
            self.logger.error(f"Failed to update threshold for {component_type}: {e}")
            return False
    
    def _validate_threshold_value(self, component_type: str, threshold: float) -> bool:
        """Validate threshold value."""
        # Basic range validation
        if not 50.0 <= threshold <= 100.0:
            return False
        
        # Component-specific validation
        component_minimums = {
            "critical_algorithms": 85.0,
            "public_apis": 80.0,
            "business_logic": 75.0,
            "integration_code": 70.0,
            "ui_components": 60.0,
            "test_utilities": 50.0
        }
        
        min_threshold = component_minimums.get(component_type, 60.0)
        return threshold >= min_threshold
    
    def _update_configuration_file(self, component_type: str, new_threshold: float) -> bool:
        """Update the configuration file with new threshold."""
        try:
            config_file = self.config_dir / "rif-workflow.yaml"
            
            # Load current configuration
            with open(config_file, 'r') as f:
                config = yaml.safe_load(f)
            
            # Update threshold
            if "quality_gates" not in config:
                config["quality_gates"] = {}
            if "coverage" not in config["quality_gates"]:
                config["quality_gates"]["coverage"] = {}
            if "thresholds" not in config["quality_gates"]["coverage"]:
                config["quality_gates"]["coverage"]["thresholds"] = {}
            
            config["quality_gates"]["coverage"]["thresholds"][component_type] = new_threshold
            
            # Save updated configuration
            with open(config_file, 'w') as f:
                yaml.dump(config, f, default_flow_style=False, indent=2)
            
            return True
            
        except Exception as e:
            self.logger.error(f"Failed to update configuration file: {e}")
            return False
    
    def _record_configuration_change(self, 
                                   component_type: str,
                                   field: str,
                                   old_value: Any,
                                   new_value: Any,
                                   reason: str,
                                   approved_by: Optional[str]) -> bool:
        """Record a configuration change for audit trail."""
        try:
            change = ConfigurationChange(
                change_id=f"change_{int(datetime.utcnow().timestamp())}",
                timestamp=datetime.utcnow().isoformat() + "Z",
                component_type=component_type,
                field=field,
                old_value=old_value,
                new_value=new_value,
                reason=reason,
                approved_by=approved_by
            )
            
            with open(self.changes_file, 'a') as f:
                f.write(json.dumps(asdict(change)) + "\n")
            
            return True
            
        except Exception as e:
            self.logger.error(f"Failed to record configuration change: {e}")
            return False
    
    def batch_update_thresholds(self, 
                               threshold_updates: Dict[str, float],
                               reason: str,
                               approved_by: Optional[str] = None) -> Dict[str, bool]:
        """
        Update multiple thresholds in batch.
        
        Args:
            threshold_updates: Dictionary of component types to new thresholds
            reason: Reason for the changes
            approved_by: Who approved the changes
            
        Returns:
            Dictionary of component types to success status
        """
        results = {}
        
        # Create single checkpoint for all changes
        checkpoint_id = f"batch_update_{int(datetime.utcnow().timestamp())}"
        self.create_checkpoint(checkpoint_id, description=f"Batch update checkpoint: {reason}")
        
        # Apply updates
        for component_type, new_threshold in threshold_updates.items():
            success = self.update_threshold(
                component_type, new_threshold, reason, approved_by, create_checkpoint_before=False
            )
            results[component_type] = success
        
        successful_updates = sum(results.values())
        self.logger.info(f"Batch update complete: {successful_updates}/{len(threshold_updates)} successful")
        
        return results
    
    def rollback_to_checkpoint(self, checkpoint_id: str) -> bool:
        """
        Rollback configuration to a previous checkpoint.
        
        Args:
            checkpoint_id: Checkpoint ID to rollback to
            
        Returns:
            True if rollback was successful
        """
        try:
            # Find checkpoint
            checkpoint = self._find_checkpoint(checkpoint_id)
            if not checkpoint:
                self.logger.error(f"Checkpoint not found: {checkpoint_id}")
                return False
            
            # Create checkpoint of current state before rollback
            rollback_checkpoint_id = f"pre_rollback_{int(datetime.utcnow().timestamp())}"
            self.create_checkpoint(rollback_checkpoint_id, description=f"Pre-rollback checkpoint before restoring {checkpoint_id}")
            
            # Restore from backup
            backup_dir = Path(checkpoint.backup_path)
            if backup_dir.exists():
                for backup_file in backup_dir.glob("*.yaml"):
                    target_file = self.config_dir / backup_file.name
                    shutil.copy2(backup_file, target_file)
                    self.logger.info(f"Restored {backup_file.name}")
            
            # Record rollback action
            self._record_configuration_change(
                "system", "rollback", "current_state", checkpoint_id, 
                f"Rollback to checkpoint {checkpoint_id}", "system"
            )
            
            self.logger.info(f"Successfully rolled back to checkpoint: {checkpoint_id}")
            return True
            
        except Exception as e:
            self.logger.error(f"Failed to rollback to checkpoint {checkpoint_id}: {e}")
            return False
    
    def _find_checkpoint(self, checkpoint_id: str) -> Optional[ConfigurationCheckpoint]:
        """Find a checkpoint by ID."""
        try:
            if not self.checkpoints_file.exists():
                return None
            
            with open(self.checkpoints_file, 'r') as f:
                for line in f:
                    try:
                        checkpoint_dict = json.loads(line.strip())
                        if checkpoint_dict["checkpoint_id"] == checkpoint_id:
                            return ConfigurationCheckpoint(**checkpoint_dict)
                    except (json.JSONDecodeError, KeyError):
                        continue
            
            return None
            
        except Exception as e:
            self.logger.error(f"Failed to find checkpoint {checkpoint_id}: {e}")
            return None
    
    def list_checkpoints(self, limit: int = 20) -> List[ConfigurationCheckpoint]:
        """
        List recent checkpoints.
        
        Args:
            limit: Maximum number of checkpoints to return
            
        Returns:
            List of ConfigurationCheckpoint objects
        """
        checkpoints = []
        
        try:
            if not self.checkpoints_file.exists():
                return checkpoints
            
            with open(self.checkpoints_file, 'r') as f:
                lines = f.readlines()
            
            # Get most recent checkpoints
            for line in reversed(lines[-limit:]):
                try:
                    checkpoint_dict = json.loads(line.strip())
                    checkpoints.append(ConfigurationCheckpoint(**checkpoint_dict))
                except (json.JSONDecodeError, TypeError):
                    continue
            
        except Exception as e:
            self.logger.error(f"Failed to list checkpoints: {e}")
        
        return checkpoints
    
    def get_change_history(self, 
                          component_type: Optional[str] = None,
                          days_back: int = 30,
                          limit: int = 50) -> List[ConfigurationChange]:
        """
        Get configuration change history.
        
        Args:
            component_type: Filter by component type (None for all)
            days_back: Number of days back to search
            limit: Maximum number of changes to return
            
        Returns:
            List of ConfigurationChange objects
        """
        changes = []
        
        try:
            if not self.changes_file.exists():
                return changes
            
            cutoff_date = datetime.utcnow() - timedelta(days=days_back)
            
            with open(self.changes_file, 'r') as f:
                lines = f.readlines()
            
            for line in reversed(lines):
                try:
                    change_dict = json.loads(line.strip())
                    
                    # Apply filters (use timezone-naive comparison)
                    change_time = datetime.fromisoformat(change_dict["timestamp"].replace('Z', ''))
                    if change_time < cutoff_date:
                        continue
                    
                    if component_type and change_dict["component_type"] != component_type:
                        continue
                    
                    changes.append(ConfigurationChange(**change_dict))
                    
                    if len(changes) >= limit:
                        break
                        
                except (json.JSONDecodeError, KeyError, ValueError):
                    continue
            
        except Exception as e:
            self.logger.error(f"Failed to get change history: {e}")
        
        return changes
    
    def validate_current_configuration(self) -> Dict[str, Any]:
        """
        Validate current configuration for consistency and correctness.
        
        Returns:
            Dictionary with validation results
        """
        validation_result = {
            "is_valid": True,
            "issues": [],
            "warnings": [],
            "configuration_summary": {}
        }
        
        try:
            # Get current thresholds
            current_thresholds = self.get_current_thresholds()
            validation_result["configuration_summary"]["thresholds"] = current_thresholds
            
            # Validate each threshold
            for component_type, threshold in current_thresholds.items():
                if not self._validate_threshold_value(component_type, threshold):
                    validation_result["is_valid"] = False
                    validation_result["issues"].append(f"Invalid threshold for {component_type}: {threshold}")
            
            # Check for missing component types
            expected_components = [
                "critical_algorithms", "public_apis", "business_logic",
                "integration_code", "ui_components", "test_utilities"
            ]
            
            for component in expected_components:
                if component not in current_thresholds:
                    validation_result["warnings"].append(f"Missing threshold configuration for {component}")
            
            # Check threshold ordering (critical should be higher than test utilities)
            if ("critical_algorithms" in current_thresholds and 
                "test_utilities" in current_thresholds):
                if current_thresholds["critical_algorithms"] <= current_thresholds["test_utilities"]:
                    validation_result["warnings"].append("Critical algorithms threshold should be higher than test utilities")
            
            # Check for recent changes that might indicate issues
            recent_changes = self.get_change_history(days_back=7)
            if len(recent_changes) > 10:
                validation_result["warnings"].append(f"High number of recent changes ({len(recent_changes)}) may indicate configuration instability")
            
            validation_result["total_issues"] = len(validation_result["issues"])
            validation_result["total_warnings"] = len(validation_result["warnings"])
            
        except Exception as e:
            validation_result["is_valid"] = False
            validation_result["issues"].append(f"Validation error: {str(e)}")
        
        return validation_result

def main():
    """Command line interface for configuration manager."""
    import argparse
    
    parser = argparse.ArgumentParser(description="Configuration Manager")
    parser.add_argument("--command", choices=["get", "update", "checkpoint", "rollback", "history", "validate"], 
                       required=True, help="Command to execute")
    parser.add_argument("--component-type", help="Component type")
    parser.add_argument("--threshold", type=float, help="New threshold value")
    parser.add_argument("--checkpoint-id", help="Checkpoint ID")
    parser.add_argument("--reason", help="Reason for change")
    
    args = parser.parse_args()
    
    manager = ConfigurationManager()
    
    if args.command == "get":
        thresholds = manager.get_current_thresholds()
        print("Current Thresholds:")
        for component, threshold in thresholds.items():
            print(f"  {component}: {threshold}%")
    
    elif args.command == "update" and args.component_type and args.threshold is not None:
        reason = args.reason or "CLI update"
        success = manager.update_threshold(args.component_type, args.threshold, reason)
        print(f"Update {'successful' if success else 'failed'}")
    
    elif args.command == "checkpoint":
        checkpoint_id = args.checkpoint_id or f"manual_{int(datetime.utcnow().timestamp())}"
        description = args.reason or "Manual checkpoint"
        success = manager.create_checkpoint(checkpoint_id, description=description)
        print(f"Checkpoint {'created' if success else 'failed'}: {checkpoint_id}")
    
    elif args.command == "rollback" and args.checkpoint_id:
        success = manager.rollback_to_checkpoint(args.checkpoint_id)
        print(f"Rollback {'successful' if success else 'failed'}")
    
    elif args.command == "history":
        changes = manager.get_change_history(args.component_type)
        print(f"Configuration Changes ({len(changes)}):")
        for change in changes[:10]:  # Show recent 10
            print(f"  {change.timestamp}: {change.component_type} {change.field} {change.old_value} -> {change.new_value}")
    
    elif args.command == "validate":
        result = manager.validate_current_configuration()
        print(f"Configuration Valid: {result['is_valid']}")
        if result["issues"]:
            print("Issues:")
            for issue in result["issues"]:
                print(f"  - {issue}")
        if result["warnings"]:
            print("Warnings:")
            for warning in result["warnings"]:
                print(f"  - {warning}")

if __name__ == "__main__":
    main()