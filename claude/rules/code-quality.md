# Code Quality Standards and Guidelines

## Core Principles

### Sandi Metz Object-Oriented Design Rules

**ALWAYS apply these principles when writing or refactoring code:**

1. **Classes can be no longer than 100 lines of code**
2. **Methods can be no longer than 5 lines of code**
3. **Pass no more than 4 parameters into a method**
4. **Controllers can instantiate only one object**

### Quality Gate Requirements

When working on code quality, always include:
- Sandi Metz object-oriented design principles
- Python PEP guidelines and coding standards
- Security coding practices and vulnerability prevention
- Performance optimization techniques
- Testing requirements and coverage goals
- Documentation standards and API design

## Automated Code Quality Tools

### Detection and Fixing

```bash
# Comprehensive code quality check
python scripts/code_quality_checker.py .

# Quick fix for common issues
python scripts/quick_fix.py

# Quick fix specific file
python scripts/quick_fix.py path/to/file.py

# Code formatting
black src/ tests/

# Linting
flake8 src/
mypy src/

# Security scanning
bandit -r src/
```

### Common Code Quality Issues

**Automatically detected and fixed:**
1. **Unused typing imports** (Tuple, Set, etc.)
2. **F-strings without placeholders** (e.g., `print(f"text")` should be `print("text")`)
3. **Type annotation issues** (parameters with `= None` but no `Optional[Type]`)
4. **Syntax errors** (unmatched parentheses, missing imports)

## Code Quality Standards

### Import Organization
- Group imports: standard library, third-party, local
- Use absolute imports from `src/`
- Remove unused imports automatically

### Type Annotations
- All functions must have type hints
- Use `Optional[Type]` for nullable parameters
- Prefer specific types over `Any`

### Error Handling
- Comprehensive error handling in all modules
- Use specific exception types
- Log errors appropriately
- Provide user-friendly error messages

### Documentation
- All modules must have docstrings
- All public functions must have docstrings
- Use Google-style docstring format
- Include examples for complex functions

## Refactoring Guidelines

When refactoring:
1. **Always include Sandi Metz principles**
2. **Load refactoring patterns and techniques**
3. **Reference code quality metrics and goals**
4. **Ensure all tests pass after refactoring**
5. **Run quality checks before committing**

## Testing Requirements

### Coverage Goals
- Minimum 90% test coverage
- All critical paths must be tested
- Integration tests for all major components
- Unit tests for all utility functions

### Quality Gates for Tests
- All tests must pass before commit
- No decrease in coverage allowed
- Performance tests for critical operations
- Security tests for sensitive operations

## Pre-commit Quality Enforcement

**Automatically prevents commits with:**
- Code formatting violations
- Type checking errors
- Security vulnerabilities
- Import conflicts
- Documentation issues

## Best Practices

### When Writing New Code
1. Check existing patterns in the codebase
2. Follow established conventions
3. Write tests first (TDD approach)
4. Document as you code
5. Run quality checks before committing

### When Fixing Issues
1. Understand the root cause
2. Write a test that reproduces the issue
3. Fix the issue
4. Verify all tests pass
5. Run quality checks
6. Document the fix

### Code Review Checklist
- [ ] Follows Sandi Metz principles
- [ ] Has appropriate test coverage
- [ ] Includes necessary documentation
- [ ] Passes all quality checks
- [ ] No security vulnerabilities
- [ ] Handles errors appropriately
- [ ] Performance is acceptable

## Integration with GitHub Issues

When working on code-quality labeled issues:
- Apply all principles in this document
- Use automated tools to find issues
- Fix issues systematically
- Document improvements made
- Update metrics in issue comments