---
name: "check-markdown-style"
description: "Check a Markdown file against docs/styleguide/markdown_styleguide.md and report violations. Use after editing `.md` files or when reviewing markdown quality."
---
# Check Markdown Style

**Version**: 4

## Goal

Check a Markdown file against the repository's authoritative Markdown style specification and report
violations with precise line references and actionable fixes.

## Inputs

* `$ARGUMENTS` - path to the Markdown file to check, relative or absolute

## Context

Read before starting:

* `docs/styleguide/markdown_styleguide.md` - the authoritative Markdown style specification

If this skill conflicts with `docs/styleguide/markdown_styleguide.md`, the styleguide wins.

## Steps

1. Read `docs/styleguide/markdown_styleguide.md` completely before checking the target file.
2. Read the target Markdown file specified in `$ARGUMENTS`.
3. Use every rule and exception in the styleguide as the checklist for the review.
4. Treat Flowmark as the canonical formatter where the styleguide says to do so. Do not flag
   Flowmark-compatible wrapping merely because it does not use the full target width.
5. For line-length review, scan every line and apply the styleguide's documented exceptions before
   reporting a violation.
6. For table review, verify the Markdown source follows the table rules in the styleguide, including
   the rule that each table row must be a single source line.
7. Report findings as a numbered list of violations. For each violation:
   * State the violated styleguide section.
   * Quote the offending line or compact line range with real line numbers.
   * Show a corrected version or a precise corrective instruction.
8. If the file has no violations, report "No style violations found."

## Output Format

Print results directly to the conversation. Use this structure:

```text
## Style Check: <filename>

### Line Length Violations

1. **Too long** (line <N>): <X> chars
   * Rule: `<styleguide section>`
   * Found: `<offending line>`
   * Fix: `<corrected version or precise corrective instruction>`

### Other Violations

1. **<Rule name>** (line <N>)
   * Rule: `<styleguide section>`
   * Found: `<offending content>`
   * Fix: `<corrected content>`

### Summary

<N> violation(s) found (<X> line-length, <Y> other).
```

## Done When

* `docs/styleguide/markdown_styleguide.md` has been read in the current turn.
* Every applicable rule from the styleguide has been checked against the file.
* Every line has been checked for line-length violations with documented exceptions applied.
* All violations are listed with line numbers, offending content, and fixes.
* A summary count is provided.

## Forbidden

* NEVER use any styleguide path other than `docs/styleguide/markdown_styleguide.md`.
* NEVER treat this skill file as a replacement for `docs/styleguide/markdown_styleguide.md`.
* NEVER fabricate line numbers; read the actual file content.
* NEVER duplicate styleguide content into this skill file.
* NEVER auto-fix the target file unless explicitly asked by the user.
* NEVER require manual reflow of already acceptable Flowmark-formatted text just because a line does
  not fully use the available width.
