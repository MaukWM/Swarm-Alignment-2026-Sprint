# Agent instructions

## Coding

Run pre-commit when done making changes:

```bash
uv run pre-commit run --all-files
```

Fix mypy issues with proper types or assertions, not `# type: ignore`.

For `pip-audit` findings: bump the version yourself when a patched release exists and nothing else has to
change. If the fix needs an implementation change, has no patched release, or would mean ignoring the
vulnerability, tell the developer.

- **Comments and docstrings**: one short line describing what the code does. No dates, ticket references,
  deployment notes or backstory.
- **No needless defensive code**: where the data contract already guarantees a value, skip `.get(x, default)`,
  `or ""` and redundant coercion. Add a guard only where the input really varies.
- **Reuse before inventing**: check whether a pattern already exists in this codebase or a reference
  implementation before writing a new one.
- **Reuse conventions**: use the existing env vars, CI variables and config profiles instead of pasting
  literal values (IP lists, secrets, endpoints) into new files.
- **Logging**: stable event names with key-value context, e.g. `logger.info("chat_created", chat_id=chat_id)`,
  `logger.error("tool_error", error=str(e))`.

## Workflow

- Keep each change to one concern; don't bundle unrelated cleanups into a task.
- Test what matters: business logic, edge cases, real input and output. Skip tests that only restate a static
  value or the type system.
