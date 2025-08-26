# Claude Code Hook Configuration Documentation

This document provides comprehensive documentation for configuring Claude Code hooks using the `.claude/settings.json` file.

## Overview

Claude Code hooks allow you to execute custom commands, scripts, or functions at specific points during a Claude Code session. This enables powerful integrations, monitoring, automation, and workflow customization.

## Hook Types

Claude Code supports the following hook types:

| Hook Type | Trigger Point | Use Cases |
|-----------|---------------|-----------|
| `SessionStart` | When a Claude Code session begins | Setup, context loading, environment preparation |
| `SessionEnd` | When a Claude Code session ends | Cleanup, logging, data export |
| `UserPromptSubmit` | When user submits a prompt | Context injection, preprocessing, logging |
| `AssistantResponse` | When assistant provides a response | Response processing, analytics, logging |
| `ToolUse` | When a tool is about to be used | Pre-execution validation, logging |
| `PostToolUse` | After a tool has been used | Post-processing, validation, cleanup |
| `ErrorOccurred` | When an error occurs | Error handling, alerting, recovery |

## Configuration Structure

### Basic Structure

```json
{
  "hooks": {
    "HookType": [
      {
        "type": "execution_type",
        "command": "shell command to execute",
        "output": "output_mode",
        "metadata": {
          "name": "Human readable name",
          "description": "What this hook does"
        }
      }
    ]
  }
}
```

### Hook Definition Properties

#### Required Properties

| Property | Type | Description | Valid Values |
|----------|------|-------------|--------------|
| `type` | string | Execution method | `command`, `script`, `function`, `webhook` |

#### Type-Specific Required Properties

Based on the `type` value, additional properties become required:

- **`type: "command"`**: Requires `command` property
- **`type: "script"`**: Requires `script` property  
- **`type: "function"`**: Requires `function` property
- **`type: "webhook"`**: Requires `url` property

#### Optional Properties

| Property | Type | Default | Description |
|----------|------|---------|-------------|
| `matcher` | string | null | Regex pattern to match trigger conditions |
| `output` | string | `"silent"` | How to handle hook output (`context`, `silent`, `stream`, `log`) |
| `timeout` | integer | 30000 | Timeout in milliseconds (100-300000) |
| `async` | boolean | false | Whether to execute asynchronously |
| `enabled` | boolean | true | Whether this hook is enabled |
| `priority` | integer | 5 | Execution priority (1=highest, 10=lowest) |
| `conditions` | object | {} | Additional execution conditions |
| `hooks` | array | [] | Nested hooks for conditional execution |
| `metadata` | object | {} | Additional metadata |

## Execution Types

### Command (`type: "command"`)

Executes a shell command.

```json
{
  "type": "command",
  "command": "gh issue list --state open --json number,title",
  "output": "context",
  "timeout": 10000
}
```

### Script (`type: "script"`)

Executes a script file.

```json
{
  "type": "script",
  "script": "./scripts/setup-workspace.sh",
  "output": "silent",
  "async": true
}
```

### Function (`type: "function"`)

Calls a predefined function (implementation dependent).

```json
{
  "type": "function",
  "function": "loadPatterns",
  "output": "context"
}
```

### Webhook (`type: "webhook"`)

Makes an HTTP request to a URL.

```json
{
  "type": "webhook",
  "url": "https://api.example.com/claude-events",
  "async": true,
  "output": "silent"
}
```

## Output Modes

| Mode | Description | Use Case |
|------|-------------|----------|
| `context` | Output becomes part of Claude's context | Providing data for Claude to use |
| `silent` | Output is discarded | Background operations |
| `stream` | Output is shown in real-time | Interactive debugging |
| `log` | Output is logged but not shown | Audit trails, monitoring |

## Conditional Execution

### Matcher Patterns

Use regular expressions to conditionally execute hooks:

```json
{
  "matcher": ".*issue.*|.*fix.*|.*implement.*",
  "hooks": [
    {
      "type": "command",
      "command": "cat ./knowledge/patterns.json",
      "output": "context"
    }
  ]
}
```

### Conditions Object

Specify additional conditions for execution:

```json
{
  "conditions": {
    "file_patterns": ["*.js", "*.ts"],
    "environment": ["DEBUG_MODE"],
    "tools": ["Edit", "Write"]
  }
}
```

## Advanced Features

### Nested Hooks

Create conditional execution chains:

```json
{
  "matcher": ".*deploy.*",
  "hooks": [
    {
      "type": "command",
      "command": "echo 'Deployment detected'",
      "output": "context",
      "hooks": [
        {
          "type": "script",
          "script": "./scripts/pre-deploy-check.sh",
          "output": "context"
        }
      ]
    }
  ]
}
```

### Priority-Based Execution

Control execution order with priorities:

```json
[
  {
    "type": "command",
    "command": "echo 'First: setup'",
    "priority": 1
  },
  {
    "type": "command", 
    "command": "echo 'Second: main task'",
    "priority": 5
  },
  {
    "type": "command",
    "command": "echo 'Last: cleanup'", 
    "priority": 10
  }
]
```

## Environment Variables

Claude Code provides environment variables that hooks can access:

| Variable | Available In | Description |
|----------|--------------|-------------|
| `CLAUDE_USER_PROMPT` | UserPromptSubmit | The user's prompt text |
| `CLAUDE_TOOL_NAME` | ToolUse, PostToolUse | Name of the tool being used |
| `CLAUDE_TOOL_EXIT_CODE` | PostToolUse | Exit code from tool execution |
| `CLAUDE_ERROR_MESSAGE` | ErrorOccurred | Error message text |
| `CLAUDE_SESSION_ID` | All hooks | Unique session identifier |
| `CLAUDE_COMMAND` | PostToolUse (Bash) | The command that was executed |

## Common Patterns

### 1. RIF Integration

```json
{
  "hooks": {
    "SessionStart": [
      {
        "type": "command",
        "command": "gh issue list --state open --label 'state:*' --json number,title,labels,body > /tmp/rif-context.json",
        "output": "context"
      }
    ],
    "PostToolUse": [
      {
        "matcher": "Edit|Write|MultiEdit",
        "hooks": [
          {
            "type": "command",
            "command": "echo '{\"timestamp\": \"'$(date -u +\"%Y-%m-%dT%H:%M:%SZ\")'\", \"action\": \"code_modified\"}' >> ./knowledge/events.jsonl",
            "output": "log"
          }
        ]
      }
    ]
  }
}
```

### 2. Development Workflow

```json
{
  "hooks": {
    "PostToolUse": [
      {
        "matcher": "Edit|Write",
        "conditions": {
          "file_patterns": ["*.js", "*.ts", "*.py"]
        },
        "hooks": [
          {
            "type": "command",
            "command": "npm run lint 2>/dev/null || flake8 . 2>/dev/null || echo 'No linter found'",
            "output": "context",
            "async": true,
            "timeout": 15000
          }
        ]
      }
    ]
  }
}
```

### 3. Error Monitoring

```json
{
  "hooks": {
    "ErrorOccurred": [
      {
        "type": "command",
        "command": "echo 'Error: $CLAUDE_ERROR_MESSAGE' >> ./logs/claude-errors.log",
        "output": "log"
      },
      {
        "type": "webhook",
        "url": "https://alerts.example.com/claude-error",
        "async": true,
        "enabled": false
      }
    ]
  }
}
```

### 4. Context Enhancement

```json
{
  "hooks": {
    "UserPromptSubmit": [
      {
        "matcher": ".*test.*|.*spec.*",
        "hooks": [
          {
            "type": "command",
            "command": "find . -name '*test*' -o -name '*spec*' | head -5",
            "output": "context"
          }
        ]
      }
    ]
  }
}
```

## Best Practices

### 1. Performance Considerations

- Use `async: true` for non-critical hooks
- Set appropriate `timeout` values
- Limit the number of hooks per type
- Use `output: "silent"` when output isn't needed

### 2. Security

- Validate all inputs in hook scripts
- Use absolute paths for script files
- Be careful with webhook URLs
- Don't expose sensitive data in hook output

### 3. Maintainability

- Always include `metadata` with `name` and `description`
- Use meaningful matcher patterns
- Group related hooks logically
- Document complex conditional logic

### 4. Testing

- Start with `enabled: false` for new hooks
- Test hooks individually before combining
- Use the validation script to check configurations
- Monitor hook execution times

## Validation

Use the provided validation script to check your configuration:

```bash
# Validate a specific configuration
python scripts/validate-hooks.py .claude/settings.json

# Validate all examples
python scripts/validate-hooks.py --examples

# Get detailed error information
python scripts/validate-hooks.py --verbose .claude/settings.json
```

## Troubleshooting

### Common Issues

1. **Hook not executing**
   - Check if `enabled: true`
   - Verify matcher pattern syntax
   - Ensure required properties are present
   - Check file permissions for scripts

2. **Timeout errors**
   - Increase `timeout` value
   - Use `async: true` for long-running tasks
   - Optimize command/script performance

3. **Output not appearing**
   - Verify `output` mode is correct
   - Check if command produces output
   - Ensure command exits successfully

4. **Schema validation errors**
   - Use the validation script to identify issues
   - Check required properties for each hook type
   - Verify JSON syntax is valid

### Debug Mode

Enable debug logging by setting environment variable:

```bash
export CLAUDE_HOOK_DEBUG=1
```

This will provide detailed logging of hook execution.

## Schema Reference

The complete JSON schema is available at `schemas/claude-hooks-schema.json`. It includes:

- Full property definitions
- Validation rules
- Type constraints
- Format specifications

## Examples

See `examples/claude-hooks-examples.json` for comprehensive examples of:

- Basic RIF integration
- Comprehensive monitoring
- Development workflows
- Minimal configurations
- Advanced conditional execution

## Migration Guide

If you have existing hook configurations, use this guide to migrate:

1. **Backup existing configuration**
2. **Run validation script** on current config
3. **Update format** as needed based on validation errors
4. **Test incrementally** by enabling hooks one by one
5. **Monitor performance** after migration

## Contributing

When contributing new hook patterns or improvements:

1. Update the schema if adding new properties
2. Add examples demonstrating new features
3. Update this documentation
4. Ensure all examples pass validation
5. Test with real Claude Code sessions