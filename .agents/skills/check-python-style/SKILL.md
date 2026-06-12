---
name: "check-python-style"
description: "Check a Python file against docs/styleguide/python_styleguide.md and report violations. Use after editing Python files or when reviewing code style."
---
# Check Python Style

**Version**: 2

## Goal

Check a Python file against the repository's authoritative Python style specification and report
violations in a way that fixes underlying problems, not just surface style.

## Inputs

* `$ARGUMENTS` - path to the Python file to check, relative or absolute

## Context

Read before starting:

* `docs/styleguide/python_styleguide.md` - the authoritative Python style specification

If this skill conflicts with `docs/styleguide/python_styleguide.md`, the styleguide wins.

## Steps

1. Read `docs/styleguide/python_styleguide.md` completely before checking the target file.
2. Read the target Python file specified in `$ARGUMENTS`.
3. Analyze the file against every rule and exception in the styleguide.
4. For each rule, check whether it actually applies in this context and honor documented
   exceptions.
5. Treat each violation as a potential mistake, not just a formatting issue. Check whether it may
   hide a correctness, typing, validation, data integrity, resource cleanup, or maintainability
   problem.
6. Report findings as a numbered list of violations. For each violation:
   * State the violated styleguide section.
   * Quote the offending line or compact line range with real line numbers.
   * Briefly state why the rule matters here.
   * Show a corrected version that fixes the underlying problem, not just the surface.
   * If a local code rewrite would be incomplete, explicitly say what surrounding code, callers, or
     downstream consumers must also be checked.
7. If a rule is stylistic only in this case, keep the explanation brief and do not invent a deeper
   problem.
8. If the file has no violations, report "No style violations found."

## Output Format

Print results directly to the conversation. Use this structure:

```text
## Style Check: <filename>

### Violations

1. **<Rule name>** (line <N>)
   * Rule: `<styleguide section>`
   * Found: `<offending code>`
   * Why this matters: `<brief risk or reason in this context>`
   * Fix: `<corrected code>`

2. ...

### Summary

<N> violation(s) found.
```

## Done When

* `docs/styleguide/python_styleguide.md` has been read in the current turn.
* Every applicable rule from the styleguide has been checked against the file.
* All violations are listed with line numbers, offending code, why they matter, and fixes.
* A summary count is provided.

## Forbidden

* NEVER use any styleguide path other than `docs/styleguide/python_styleguide.md`.
* NEVER treat this skill file as a replacement for `docs/styleguide/python_styleguide.md`.
* NEVER skip rules; check the file against the complete styleguide.
* NEVER ignore rule-specific exceptions from the styleguide.
* NEVER fabricate line numbers; read the actual file content.
* NEVER duplicate styleguide content into this skill file.
* NEVER suggest a surface-only rewrite when the rule violation points to a deeper problem in types,
  validation, data meaning, callers, or cleanup logic.
* NEVER auto-fix the target file unless explicitly asked by the user.
