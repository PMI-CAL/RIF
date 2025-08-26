#!/usr/bin/env python3
"""
Path Resolution System for RIF Deployment Configuration
Provides portable, environment-agnostic path resolution
"""

import os
import json
import logging
from pathlib import Path
from typing import Dict, Optional, Union, Any

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class PathResolver:
    """
    Resolves configuration paths with variable substitution
    Supports PROJECT_ROOT, HOME, and custom environment variables
    """
    
    def __init__(self, config_path: Optional[Union[str, Path]] = None):
        """
        Initialize path resolver with configuration
        
        Args:
            config_path: Optional path to deploy.config.json file
        """
        self.project_root = self._find_project_root()
        self.config = self._load_config(config_path)
        self._setup_environment_variables()
        
        logger.info(f"Path resolver initialized with project root: {self.project_root}")
    
    def resolve(self, path_key: str) -> Path:
        """
        Resolve a path key to absolute path
        
        Args:
            path_key: Key from the paths section of config
            
        Returns:
            Resolved absolute Path object
            
        Raises:
            KeyError: If path_key not found in configuration
        """
        if path_key not in self.config.get('paths', {}):
            raise KeyError(f"Path key '{path_key}' not found in configuration")
        
        template = self.config['paths'][path_key]
        resolved_path = self._expand_variables(template)
        
        logger.debug(f"Resolved '{path_key}': {template} -> {resolved_path}")
        
        return resolved_path
    
    def get_config(self, section: Optional[str] = None) -> Dict[str, Any]:
        """
        Get configuration section or entire config
        
        Args:
            section: Optional configuration section name
            
        Returns:
            Configuration dictionary or section
        """
        if section is None:
            return self.config
        return self.config.get(section, {})
    
    def is_feature_enabled(self, feature_name: str) -> bool:
        """
        Check if a feature is enabled in configuration
        
        Args:
            feature_name: Name of the feature to check
            
        Returns:
            True if feature is enabled, False otherwise
        """
        return self.config.get('features', {}).get(feature_name, False)
    
    def get_deployment_mode(self) -> str:
        """
        Get current deployment mode
        
        Returns:
            Deployment mode string (e.g., 'project', 'development')
        """
        return self.config.get('deployment_mode', 'project')
    
    def _find_project_root(self) -> Path:
        """
        Find project root by looking for .rif directory or deploy.config.json
        
        Returns:
            Path to project root directory
        """
        current = Path.cwd()
        
        # Look for indicators of RIF project root
        indicators = ['.rif', 'deploy.config.json', 'CLAUDE.md', '.git']
        
        while current != current.parent:
            for indicator in indicators:
                if (current / indicator).exists():
                    logger.debug(f"Found project root indicator '{indicator}' at {current}")
                    return current
            current = current.parent
        
        # If no indicators found, use current working directory
        logger.warning(f"No project root indicators found, using current directory: {Path.cwd()}")
        return Path.cwd()
    
    def _load_config(self, config_path: Optional[Union[str, Path]] = None) -> Dict[str, Any]:
        """
        Load deployment configuration from JSON file
        
        Args:
            config_path: Optional path to configuration file
            
        Returns:
            Configuration dictionary
        """
        if config_path is None:
            config_path = self.project_root / 'deploy.config.json'
        else:
            config_path = Path(config_path)
        
        try:
            if config_path.exists():
                with open(config_path, 'r') as f:
                    config = json.load(f)
                logger.info(f"Loaded configuration from {config_path}")
                return config
            else:
                logger.warning(f"Configuration file not found at {config_path}, using defaults")
                return self._get_default_config()
        except (json.JSONDecodeError, IOError) as e:
            logger.error(f"Error loading configuration from {config_path}: {e}")
            return self._get_default_config()
    
    def _get_default_config(self) -> Dict[str, Any]:
        """
        Get default configuration if no config file is found
        
        Returns:
            Default configuration dictionary
        """
        return {
            "version": "1.0.0",
            "deployment_mode": "project",
            "paths": {
                "rif_home": "${PROJECT_ROOT}/.rif",
                "knowledge_base": "${PROJECT_ROOT}/.rif/knowledge",
                "agents": "${PROJECT_ROOT}/.rif/agents",
                "commands": "${PROJECT_ROOT}/.rif/commands",
                "docs": "${PROJECT_ROOT}/docs",
                "config": "${PROJECT_ROOT}/config",
                "scripts": "${PROJECT_ROOT}/scripts"
            },
            "features": {
                "self_development_checks": False,
                "audit_logging": False,
                "development_telemetry": False
            },
            "knowledge": {
                "preserve_patterns": True,
                "preserve_decisions": False,
                "clean_on_init": True
            }
        }
    
    def _setup_environment_variables(self):
        """
        Setup environment variables from configuration
        """
        # Set PROJECT_ROOT if not already set
        if 'PROJECT_ROOT' not in os.environ:
            os.environ['PROJECT_ROOT'] = str(self.project_root)
        
        # Set RIF_HOME if not already set
        if 'RIF_HOME' not in os.environ:
            os.environ['RIF_HOME'] = str(self.project_root / '.rif')
        
        # Set deployment mode
        os.environ['RIF_DEPLOYMENT_MODE'] = self.get_deployment_mode()
    
    def _expand_variables(self, template: str) -> Path:
        """
        Expand environment variables in path templates
        
        Args:
            template: Path template with variables
            
        Returns:
            Expanded Path object
        """
        # Define variable replacements
        replacements = {
            '${PROJECT_ROOT}': str(self.project_root),
            '${HOME}': os.environ.get('HOME', str(Path.home())),
            '${RIF_HOME}': os.environ.get('RIF_HOME', str(self.project_root / '.rif')),
            '${USER}': os.environ.get('USER', 'unknown')
        }
        
        # Replace variables in template
        expanded = template
        for variable, value in replacements.items():
            expanded = expanded.replace(variable, value)
        
        # Handle remaining environment variables
        expanded = os.path.expandvars(expanded)
        
        return Path(expanded).resolve()

class ConfigurationValidator:
    """
    Validates deployment configuration for correctness and security
    """
    
    def __init__(self, resolver: PathResolver):
        self.resolver = resolver
    
    def validate(self) -> tuple[bool, list[str]]:
        """
        Validate the configuration
        
        Returns:
            Tuple of (is_valid, list_of_errors)
        """
        errors = []
        config = self.resolver.get_config()
        
        # Validate required sections
        required_sections = ['version', 'deployment_mode', 'paths']
        for section in required_sections:
            if section not in config:
                errors.append(f"Missing required configuration section: {section}")
        
        # Validate paths exist or can be created
        if 'paths' in config:
            for path_key, path_template in config['paths'].items():
                try:
                    resolved_path = self.resolver.resolve(path_key)
                    parent_dir = resolved_path.parent
                    
                    # Check if parent directory exists or can be created
                    if not parent_dir.exists():
                        try:
                            parent_dir.mkdir(parents=True, exist_ok=True)
                        except PermissionError:
                            errors.append(f"Cannot create directory for path '{path_key}': {parent_dir}")
                except Exception as e:
                    errors.append(f"Error resolving path '{path_key}': {e}")
        
        # Validate deployment mode
        valid_modes = ['project', 'development', 'production']
        deployment_mode = config.get('deployment_mode')
        if deployment_mode not in valid_modes:
            errors.append(f"Invalid deployment_mode '{deployment_mode}'. Must be one of: {valid_modes}")
        
        return len(errors) == 0, errors

# Utility functions for common use cases
def get_default_resolver() -> PathResolver:
    """
    Get a default path resolver instance
    
    Returns:
        PathResolver instance with default configuration
    """
    return PathResolver()

def resolve_rif_path(path_key: str) -> Path:
    """
    Convenience function to resolve a RIF path
    
    Args:
        path_key: Path key to resolve
        
    Returns:
        Resolved Path object
    """
    resolver = get_default_resolver()
    return resolver.resolve(path_key)

def validate_configuration() -> tuple[bool, list[str]]:
    """
    Validate the current deployment configuration
    
    Returns:
        Tuple of (is_valid, list_of_errors)
    """
    resolver = get_default_resolver()
    validator = ConfigurationValidator(resolver)
    return validator.validate()

if __name__ == "__main__":
    # Command-line interface for testing
    import sys
    
    resolver = get_default_resolver()
    
    if len(sys.argv) > 1:
        path_key = sys.argv[1]
        try:
            resolved_path = resolver.resolve(path_key)
            print(f"{path_key}: {resolved_path}")
        except KeyError as e:
            print(f"Error: {e}")
            sys.exit(1)
    else:
        print("Available path keys:")
        for key in resolver.get_config('paths').keys():
            try:
                resolved = resolver.resolve(key)
                print(f"  {key}: {resolved}")
            except Exception as e:
                print(f"  {key}: ERROR - {e}")
        
        print(f"\nProject root: {resolver.project_root}")
        print(f"Deployment mode: {resolver.get_deployment_mode()}")
        
        # Validate configuration
        is_valid, errors = validate_configuration()
        print(f"\nConfiguration valid: {is_valid}")
        if errors:
            print("Validation errors:")
            for error in errors:
                print(f"  - {error}")