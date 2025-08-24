# Claude Code Hook Configuration Schema

This directory contains a comprehensive JSON schema for validating Claude Code hook configurations, along with examples, validation tools, and documentation.

## Quick Start

1. **Validate your configuration:**
   ```bash
   python3 scripts/validate-hooks.py .claude/settings.json
   ```

2. **See example configurations:**
   ```bash
   cat examples/claude-hooks-examples.json
   ```

3. **Read the documentation:**
   ```bash
   cat docs/claude-hooks-documentation.md
   ```

## Files Overview

| File | Description |
|------|-------------|
| `schemas/claude-hooks-schema.json` | Complete JSON schema for hook validation |
| `examples/claude-hooks-examples.json` | Working examples for all hook types |
| `scripts/validate-hooks.py` | Python validation script with detailed error reporting |
| `docs/claude-hooks-documentation.md` | Comprehensive documentation and best practices |
| `README-hooks.md` | This file - quick overview and usage guide |

## Supported Hook Types

- **SessionStart** - Execute when Claude Code session begins
- **SessionEnd** - Execute when Claude Code session ends  
- **UserPromptSubmit** - Execute when user submits a prompt
- **AssistantResponse** - Execute when assistant provides a response
- **ToolUse** - Execute when a tool is about to be used
- **PostToolUse** - Execute after a tool has been used
- **ErrorOccurred** - Execute when an error occurs

## Execution Types

- **command** - Run shell commands
- **script** - Execute script files
- **function** - Call predefined functions
- **webhook** - Make HTTP requests

## Validation

The validation script provides:
- JSON schema validation
- Detailed error messages with suggestions
- Configuration summaries
- Best practice warnings

```bash
# Validate specific file
python3 scripts/validate-hooks.py .claude/settings.json

# Validate all examples
python3 scripts/validate-hooks.py --examples

# Get verbose output
python3 scripts/validate-hooks.py --verbose .claude/settings.json
```

## Example Usage

### Basic RIF Integration
```json
{
  "hooks": {
    "SessionStart": [
      {
        "type": "command",
        "command": "gh issue list --state open --json number,title",
        "output": "context"
      }
    ]
  }
}
```

### Conditional Execution
```json
{
  "hooks": {
    "UserPromptSubmit": [
      {
        "type": "command",
        "command": "cat ./patterns.json",
        "matcher": ".*implement.*|.*fix.*",
        "output": "context"
      }
    ]
  }
}
```

### Development Workflow
```json
{
  "hooks": {
    "PostToolUse": [
      {
        "type": "command",
        "command": "npm run lint",
        "matcher": "Edit|Write",
        "output": "context",
        "async": true,
        "conditions": {
          "file_patterns": ["*.js", "*.ts"]
        }
      }
    ]
  }
}
```

## Requirements

- Python 3.6+
- `jsonschema` library: `pip install jsonschema`

## Implementation Status

✅ **Complete** - All deliverables implemented:
- [x] JSON schema for hook validation
- [x] Example configurations for all hook types  
- [x] Validation script with error reporting
- [x] Comprehensive documentation
- [x] All examples pass validation
- [x] Schema handles conditional execution, nested hooks, and all specified features

## Testing

All examples have been validated against the schema:
```bash
$ python3 scripts/validate-hooks.py --examples
Validating 5 example configurations...
✅ All configurations are valid!
```

## Integration

This schema is designed to work with Claude Code's existing hook system and is compatible with the RIF (Reactive Intelligence Framework) patterns shown in the CLAUDE.md file.

To use in your project:
1. Copy the schema file to your project
2. Reference it in your IDE for autocompletion
3. Use the validation script in CI/CD pipelines
4. Follow the examples for common use cases