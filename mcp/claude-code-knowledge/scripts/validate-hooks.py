#!/usr/bin/env python3
"""
Claude Code Hooks Configuration Validator

This script validates Claude Code hook configurations against the JSON schema.
It can validate both individual configuration files and the examples provided.

Usage:
    python validate-hooks.py [config-file]
    python validate-hooks.py --examples
    python validate-hooks.py --help

Examples:
    python validate-hooks.py .claude/settings.json
    python validate-hooks.py --examples
"""

import json
import os
import sys
import argparse
from pathlib import Path
from typing import Dict, List, Any, Tuple

try:
    import jsonschema
    from jsonschema import validate, ValidationError, Draft7Validator
except ImportError:
    print("Error: jsonschema library not found.")
    print("Install it with: pip install jsonschema")
    sys.exit(1)


class HookValidator:
    """Validates Claude Code hook configurations against the JSON schema."""
    
    def __init__(self, schema_path: str = None):
        """Initialize the validator with the schema."""
        if schema_path is None:
            # Default to schema in the same repository
            script_dir = Path(__file__).parent
            schema_path = script_dir.parent / "schemas" / "claude-hooks-schema.json"
        
        self.schema_path = Path(schema_path)
        self.schema = self._load_schema()
        self.validator = Draft7Validator(self.schema)
    
    def _load_schema(self) -> Dict[str, Any]:
        """Load the JSON schema from file."""
        try:
            with open(self.schema_path, 'r') as f:
                return json.load(f)
        except FileNotFoundError:
            print(f"Error: Schema file not found at {self.schema_path}")
            sys.exit(1)
        except json.JSONDecodeError as e:
            print(f"Error: Invalid JSON in schema file: {e}")
            sys.exit(1)
    
    def validate_config(self, config: Dict[str, Any]) -> Tuple[bool, List[str]]:
        """
        Validate a configuration against the schema.
        
        Returns:
            Tuple of (is_valid, list_of_errors)
        """
        errors = []
        
        try:
            validate(config, self.schema)
            return True, []
        except ValidationError as e:
            errors.append(f"Validation error: {e.message}")
            if e.path:
                errors.append(f"  Path: {' -> '.join(str(p) for p in e.path)}")
            return False, errors
    
    def validate_file(self, file_path: str) -> Tuple[bool, List[str]]:
        """Validate a configuration file."""
        try:
            with open(file_path, 'r') as f:
                config = json.load(f)
        except FileNotFoundError:
            return False, [f"File not found: {file_path}"]
        except json.JSONDecodeError as e:
            return False, [f"Invalid JSON in {file_path}: {e}"]
        
        return self.validate_config(config)
    
    def get_detailed_errors(self, config: Dict[str, Any]) -> List[str]:
        """Get detailed validation errors with suggestions."""
        detailed_errors = []
        
        for error in self.validator.iter_errors(config):
            error_msg = f"Error at {' -> '.join(str(p) for p in error.path)}: {error.message}"
            detailed_errors.append(error_msg)
            
            # Add helpful suggestions for common errors
            if "required" in error.message.lower():
                detailed_errors.append("  Suggestion: Check that all required fields are present")
            elif "enum" in error.message.lower():
                if error.schema.get("enum"):
                    valid_values = ", ".join(str(v) for v in error.schema["enum"])
                    detailed_errors.append(f"  Valid values: {valid_values}")
            elif "format" in error.message.lower():
                detailed_errors.append("  Suggestion: Check the format of the field (e.g., URI format for URLs)")
        
        return detailed_errors


def validate_examples(validator: HookValidator) -> bool:
    """Validate all example configurations."""
    script_dir = Path(__file__).parent
    examples_path = script_dir.parent / "examples" / "claude-hooks-examples.json"
    
    try:
        with open(examples_path, 'r') as f:
            examples_data = json.load(f)
    except (FileNotFoundError, json.JSONDecodeError) as e:
        print(f"Error loading examples: {e}")
        return False
    
    all_valid = True
    examples = examples_data.get("examples", {})
    
    print(f"Validating {len(examples)} example configurations...")
    print("=" * 60)
    
    for example_name, example_data in examples.items():
        config = example_data.get("config", {})
        description = example_data.get("description", "No description")
        
        print(f"\n📋 Example: {example_name}")
        print(f"Description: {description}")
        
        is_valid, errors = validator.validate_config(config)
        
        if is_valid:
            print("✅ Valid configuration")
        else:
            print("❌ Invalid configuration")
            all_valid = False
            for error in errors:
                print(f"  {error}")
            
            # Show detailed errors for debugging
            detailed_errors = validator.get_detailed_errors(config)
            if detailed_errors:
                print("  Detailed errors:")
                for error in detailed_errors:
                    print(f"    {error}")
    
    return all_valid


def main():
    """Main function to handle command line arguments and run validation."""
    parser = argparse.ArgumentParser(
        description="Validate Claude Code hook configurations",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  %(prog)s .claude/settings.json    # Validate a specific config file
  %(prog)s --examples              # Validate all example configurations
  %(prog)s --schema custom.json config.json  # Use custom schema
        """
    )
    
    parser.add_argument(
        'config_file',
        nargs='?',
        help='Path to the configuration file to validate'
    )
    parser.add_argument(
        '--examples',
        action='store_true',
        help='Validate all example configurations'
    )
    parser.add_argument(
        '--schema',
        help='Path to custom schema file (default: ../schemas/claude-hooks-schema.json)'
    )
    parser.add_argument(
        '--verbose',
        '-v',
        action='store_true',
        help='Show detailed error information'
    )
    
    args = parser.parse_args()
    
    # Create validator
    validator = HookValidator(args.schema)
    
    if args.examples:
        # Validate examples
        success = validate_examples(validator)
        sys.exit(0 if success else 1)
    
    elif args.config_file:
        # Validate specific file
        print(f"Validating configuration file: {args.config_file}")
        print("=" * 60)
        
        is_valid, errors = validator.validate_file(args.config_file)
        
        if is_valid:
            print("✅ Configuration is valid!")
            
            # Additional checks for best practices
            try:
                with open(args.config_file, 'r') as f:
                    config = json.load(f)
                
                hooks = config.get("hooks", {})
                total_hooks = sum(len(hook_list) for hook_list in hooks.values())
                print(f"📊 Configuration summary:")
                print(f"  - Hook types: {len(hooks)}")
                print(f"  - Total hooks: {total_hooks}")
                
                if total_hooks == 0:
                    print("⚠️  Warning: No hooks defined in configuration")
                elif total_hooks > 20:
                    print("⚠️  Warning: Large number of hooks may impact performance")
                    
            except Exception:
                pass  # Ignore errors in summary generation
                
        else:
            print("❌ Configuration is invalid!")
            for error in errors:
                print(f"  {error}")
            
            if args.verbose:
                try:
                    with open(args.config_file, 'r') as f:
                        config = json.load(f)
                    detailed_errors = validator.get_detailed_errors(config)
                    if detailed_errors:
                        print("\nDetailed errors:")
                        for error in detailed_errors:
                            print(f"  {error}")
                except Exception:
                    pass
        
        sys.exit(0 if is_valid else 1)
    
    else:
        # No arguments provided
        parser.print_help()
        sys.exit(1)


if __name__ == "__main__":
    main()