# NXS Formatting Standard

This document defines the official formatting rules for Nemantix script files.

## Applicability

All rules apply to `.nxs`, `.nxc`, and `.nxv` files. The formatter behavior differs
by file type and invocation flag:

| Extension | Description                       | Default behaviour                                                                         | `--check`   |
|-----------|-----------------------------------|-------------------------------------------------------------------------------------------|-------------|
| `.nxs`    | NXS source files                  | reformat                                                                                  | report only |
| `.nxc`    | Compiled executable specification | reformat                                                                                  | report only |
| `.nxv`    | Signed and verifiable NXC         | **report only** (complain mode — modifying the file would invalidate the ECDSA signature) | report only |

Rules are identified by the prefix **NXF** (Nemantix NXS Formatting) followed by a three-digit code.

A compliant formatter must enforce all rules marked **enforced**. Rules marked **strict** are
only applied when the formatter is invoked with the `--strict` flag.

## Rule Index

### NXF0xx — General Structure

| Code                           | Summary                                          | Mode     |
|--------------------------------|--------------------------------------------------|----------|
| [NXF001](formatting/NXF001.md) | Use 2 spaces per indentation level               | enforced |
| [NXF002](formatting/NXF002.md) | Lines must not exceed 120 characters             | enforced |
| [NXF003](formatting/NXF003.md) | Exactly one blank line between top-level blocks  | enforced |
| [NXF004](formatting/NXF004.md) | Exactly one blank line between internal sections | enforced |
| [NXF005](formatting/NXF005.md) | No blank line immediately before a block closer  | enforced |

### NXF1xx — Blocks and Closers

| Code                           | Summary                                 | Mode   |
|--------------------------------|-----------------------------------------|--------|
| [NXF101](formatting/NXF101.md) | Block closers must use the specific tag | strict |

### NXF2xx — Prompts

| Code                           | Summary                                                | Mode     |
|--------------------------------|--------------------------------------------------------|----------|
| [NXF201](formatting/NXF201.md) | Use inline prompt when line fits within 120 characters | enforced |

### NXF4xx — Imports and `do` Statements

| Code                           | Summary                                                      | Mode     |
|--------------------------------|--------------------------------------------------------------|----------|
| [NXF401](formatting/NXF401.md) | Prefer inline import form when it fits within 120 characters | enforced |
| [NXF402](formatting/NXF402.md) | Prefer inline `do` form when it fits within 120 characters   | enforced |

### NXF5xx — Annotations and Intentable Prefix

| Code                           | Summary                                                   | Mode     |
|--------------------------------|-----------------------------------------------------------|----------|
| [NXF501](formatting/NXF501.md) | Annotations are indented at the same level as their block | enforced |

---

## Formatter Modes

**Default mode** — enforces all rules not marked strict. Tolerates existing style choices
that do not violate enforced rules (e.g. generic `__` closers are left as-is).

**Strict mode (`--strict`)** — additionally applies strict rules, normalising the entire
file to the canonical form.

**Check mode (`--check`)** — reports all violations without modifying any file. Exits
with a non-zero status code if any violation is found. Automatically active for `.nxv`
files regardless of other flags.