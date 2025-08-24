"""
Configuration management for Claude Code Knowledge MCP Server.

Provides centralized configuration management with environment variable support,
validation, and defaults. Integrates with existing RIF configuration system.
"""

import os
import json
import logging
from typing import Dict, Any, Optional
from pathlib import Path
from dataclasses import dataclass


@dataclass
class ServerConfig:
    """Server configuration with validation and defaults."""
    
    # Logging configuration
    log_level: str = "INFO"
    log_format: str = "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
    
    # Performance configuration  
    cache_size: int = 100
    cache_ttl: int = 300  # 5 minutes
    timeout_seconds: int = 30
    max_request_size_mb: int = 1
    target_response_time_ms: int = 200
    max_concurrent_requests: int = 10
    
    # Database configuration
    database_path: str = "knowledge/hybrid_knowledge.duckdb"
    connection_pool_size: int = 5
    query_timeout_seconds: int = 10
    
    # Security configuration
    input_validation: bool = True
    output_sanitization: bool = True
    rate_limit_requests_per_minute: int = 60
    rate_limit_burst_size: int = 10
    
    # Feature flags
    enable_caching: bool = True
    enable_metrics: bool = True
    enable_debug_mode: bool = False
    enable_query_logging: bool = False
    
    # Integration settings
    rif_root_path: str = ""
    knowledge_seeding_required: bool = True
    graceful_degradation: bool = True
    
    def __post_init__(self):
        """Validate configuration values."""
        if self.cache_size < 10 or self.cache_size > 1000:
            raise ValueError("cache_size must be between 10 and 1000")
            
        if self.cache_ttl < 60 or self.cache_ttl > 3600:
            raise ValueError("cache_ttl must be between 60 and 3600 seconds")
            
        if self.timeout_seconds < 5 or self.timeout_seconds > 120:
            raise ValueError("timeout_seconds must be between 5 and 120")
            
        if self.log_level not in ["DEBUG", "INFO", "WARNING", "ERROR"]:
            raise ValueError("log_level must be DEBUG, INFO, WARNING, or ERROR")


class ConfigManager:
    """Manages configuration loading and environment integration."""
    
    def __init__(self, config_path: Optional[str] = None):
        self.config_path = config_path
        self.logger = logging.getLogger(__name__)
        
    def load_config(self) -> ServerConfig:
        """Load configuration from multiple sources with precedence."""
        config_data = {}
        
        # 1. Load from default file if it exists
        default_config_path = Path(__file__).parent / "config.json"
        if default_config_path.exists():
            config_data.update(self._load_json_config(default_config_path))
        
        # 2. Load from specified config file
        if self.config_path and Path(self.config_path).exists():
            config_data.update(self._load_json_config(self.config_path))
        
        # 3. Load from environment variables (highest precedence)
        config_data.update(self._load_env_config())
        
        # 4. Apply RIF-specific defaults
        config_data.update(self._get_rif_defaults())
        
        return ServerConfig(**config_data)
    
    def _load_json_config(self, config_path: Path) -> Dict[str, Any]:
        """Load configuration from JSON file."""
        try:
            with open(config_path, 'r') as f:
                config = json.load(f)
                self.logger.info(f"Loaded configuration from {config_path}")
                return config
        except Exception as e:
            self.logger.warning(f"Failed to load config from {config_path}: {e}")
            return {}
    
    def _load_env_config(self) -> Dict[str, Any]:
        """Load configuration from environment variables."""
        config = {}
        
        # Environment variable mapping
        env_mappings = {
            'MCP_LOG_LEVEL': 'log_level',
            'MCP_CACHE_SIZE': ('cache_size', int),
            'MCP_CACHE_TTL': ('cache_ttl', int),
            'MCP_TIMEOUT': ('timeout_seconds', int),
            'MCP_MAX_REQUEST_SIZE': ('max_request_size_mb', int),
            'MCP_DB_PATH': 'database_path',
            'MCP_ENABLE_CACHING': ('enable_caching', bool),
            'MCP_ENABLE_DEBUG': ('enable_debug_mode', bool),
            'MCP_RATE_LIMIT_RPM': ('rate_limit_requests_per_minute', int),
            'RIF_ROOT': 'rif_root_path'
        }
        
        for env_var, config_key in env_mappings.items():
            env_value = os.getenv(env_var)
            if env_value is not None:
                if isinstance(config_key, tuple):
                    key, converter = config_key
                    try:
                        if converter == bool:
                            config[key] = env_value.lower() in ('true', '1', 'yes', 'on')
                        else:
                            config[key] = converter(env_value)
                    except ValueError as e:
                        self.logger.warning(f"Invalid value for {env_var}: {env_value} ({e})")
                else:
                    config[config_key] = env_value
        
        if config:
            self.logger.info(f"Loaded {len(config)} configuration values from environment")
        
        return config
    
    def _get_rif_defaults(self) -> Dict[str, Any]:
        """Get RIF-specific configuration defaults."""
        # Try to determine RIF root path
        rif_root = self._find_rif_root()
        
        defaults = {
            'rif_root_path': rif_root,
            'database_path': os.path.join(rif_root, 'knowledge/hybrid_knowledge.duckdb') if rif_root else 'knowledge/hybrid_knowledge.duckdb'
        }
        
        # Check if we're in a development environment
        if rif_root and Path(rif_root, '.git').exists():
            defaults.update({
                'enable_debug_mode': True,
                'enable_query_logging': True,
                'log_level': 'DEBUG'
            })
        
        return defaults
    
    def _find_rif_root(self) -> str:
        """Find the RIF root directory."""
        # Start from current file location and walk up
        current = Path(__file__).parent
        
        for _ in range(5):  # Maximum 5 levels up
            if (current / 'knowledge').exists() and (current / 'claude').exists():
                return str(current)
            current = current.parent
        
        # Fallback to environment variable or current working directory
        return os.getenv('RIF_ROOT', os.getcwd())
    
    def save_config(self, config: ServerConfig, output_path: Optional[str] = None) -> bool:
        """Save configuration to file."""
        try:
            output_file = output_path or (Path(__file__).parent / "config.json")
            
            # Convert dataclass to dict
            config_dict = {
                key: value for key, value in config.__dict__.items()
                if not key.startswith('_')
            }
            
            with open(output_file, 'w') as f:
                json.dump(config_dict, f, indent=2)
            
            self.logger.info(f"Configuration saved to {output_file}")
            return True
            
        except Exception as e:
            self.logger.error(f"Failed to save configuration: {e}")
            return False
    
    def validate_config(self, config: ServerConfig) -> bool:
        """Validate configuration against requirements."""
        validation_errors = []
        
        # Check database path exists
        if config.rif_root_path:
            db_path = Path(config.rif_root_path) / config.database_path
            if not db_path.exists():
                validation_errors.append(f"Database file not found: {db_path}")
        
        # Check RIF knowledge seeding
        if config.knowledge_seeding_required and config.rif_root_path:
            seed_file = Path(config.rif_root_path) / "knowledge/schema/seed_claude_knowledge.py"
            if not seed_file.exists():
                validation_errors.append("Claude knowledge seeding file not found")
        
        # Log validation results
        if validation_errors:
            for error in validation_errors:
                self.logger.error(f"Configuration validation error: {error}")
            return False
        else:
            self.logger.info("Configuration validation passed")
            return True


def create_default_config_file():
    """Create a default configuration file."""
    default_config = {
        "log_level": "INFO",
        "cache_size": 100,
        "cache_ttl": 300,
        "timeout_seconds": 30,
        "max_request_size_mb": 1,
        "target_response_time_ms": 200,
        "max_concurrent_requests": 10,
        "database_path": "knowledge/hybrid_knowledge.duckdb",
        "connection_pool_size": 5,
        "query_timeout_seconds": 10,
        "input_validation": True,
        "output_sanitization": True,
        "rate_limit_requests_per_minute": 60,
        "rate_limit_burst_size": 10,
        "enable_caching": True,
        "enable_metrics": True,
        "enable_debug_mode": False,
        "enable_query_logging": False,
        "knowledge_seeding_required": True,
        "graceful_degradation": True
    }
    
    config_file = Path(__file__).parent / "config.json"
    
    try:
        with open(config_file, 'w') as f:
            json.dump(default_config, f, indent=2)
        print(f"Default configuration created at {config_file}")
        return True
    except Exception as e:
        print(f"Failed to create default configuration: {e}")
        return False


def load_server_config(config_path: Optional[str] = None) -> ServerConfig:
    """Convenience function to load server configuration."""
    manager = ConfigManager(config_path)
    config = manager.load_config()
    
    # Validate the configuration
    if not manager.validate_config(config):
        logging.warning("Configuration validation failed - some features may not work properly")
    
    return config


if __name__ == '__main__':
    """Command line interface for configuration management."""
    import argparse
    
    parser = argparse.ArgumentParser(description='Claude Code Knowledge MCP Server Configuration')
    parser.add_argument('--create-default', action='store_true', help='Create default configuration file')
    parser.add_argument('--validate', type=str, help='Validate configuration file')
    parser.add_argument('--show-config', action='store_true', help='Show current configuration')
    
    args = parser.parse_args()
    
    if args.create_default:
        create_default_config_file()
    elif args.validate:
        manager = ConfigManager(args.validate)
        config = manager.load_config()
        is_valid = manager.validate_config(config)
        print(f"Configuration is {'valid' if is_valid else 'invalid'}")
    elif args.show_config:
        config = load_server_config()
        print(json.dumps(config.__dict__, indent=2))
    else:
        parser.print_help()