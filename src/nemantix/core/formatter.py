from __future__ import annotations

import difflib

from nemantix.core.formatting import NXFViolation, apply_edits
from nemantix.core.formatting._edit import NXFEdit
from nemantix.core.formatting._helpers import indent_level, is_closer, is_section_header
from nemantix.core.formatting.rules import ALL_RULES
from nemantix.core.parser import ParserLark


class Formatter:
    """Formatter for Nemantix source files.

    format() parses the source into an AST and re-serializes it, so any
    indentation error (wrong unit, missing levels, flat output) is corrected
    structurally.  check() runs each registered NXFRule against the source
    and returns all violations without modifying it.

    strict=True enforces NXF101: all closers are emitted in their explicit form
    (__body, __action, etc.).  strict=False (default) preserves the closer style
    used in the original source via order-based mapping.
    """

    def __init__(self, strict: bool = True) -> None:
        self._strict = strict

    def format(self, source: str) -> str:
        # AST round-trip: fixes NXF001, NXF101, NXF201, NXF401, NXF502, NXF003/004/005.
        stmts = ParserLark().parse(source, "<formatter>")
        serialized = "\n\n".join(stmt.to_nxs() for stmt in stmts)
        lines: list[str] = list(serialized.splitlines())
        lines = self._enforce_section_blank_lines(lines)
        result = "\n".join(lines)
        if source.endswith("\n"):
            result += "\n"
        if not self._strict:
            result = self._restore_closer_style(source, result)

        # Text-level rule fixes (e.g. NXF402: block do → inline).
        # Re-parse so FileMeta offsets match the round-tripped source.
        stmts2 = ParserLark().parse(result, "<formatter>")
        fixes = [
            v.fix
            for rule in ALL_RULES
            for v in rule.detect(stmts2, result.splitlines())
            if v.fix is not None
        ]
        if fixes:
            result = apply_edits(result, fixes)

        return result

    def check(self, source: str) -> list[NXFViolation]:
        stmts = ParserLark().parse(source, "<formatter>")
        lines = source.splitlines()
        violations: list[NXFViolation] = []
        for rule in ALL_RULES:
            violations.extend(rule.detect(stmts, lines))
        existing_nxf001 = {v.line for v in violations if v.rule == "NXF001"}
        violations.extend(
            self._structural_indent_violations(source, lines, stmts, existing_nxf001)
        )
        return violations

    def _structural_indent_violations(
        self, source: str, orig_lines: list[str], stmts: list, existing_nxf001: set[int]
    ) -> list[NXFViolation]:
        """Detect lines where the AST round-trip corrects only the indentation level."""
        serialized = "\n\n".join(stmt.to_nxs() for stmt in stmts)
        rt_lines = self._enforce_section_blank_lines(list(serialized.splitlines()))
        if not self._strict:
            rt_source = self._restore_closer_style(source, "\n".join(rt_lines))
            rt_lines = rt_source.splitlines()
        violations: list[NXFViolation] = []
        matcher = difflib.SequenceMatcher(None, orig_lines, rt_lines, autojunk=False)
        for tag, i1, i2, j1, j2 in matcher.get_opcodes():
            if tag != "replace" or (i2 - i1) != (j2 - j1):
                continue
            for di in range(i2 - i1):
                orig = orig_lines[i1 + di]
                expected = rt_lines[j1 + di]
                if orig.lstrip() != expected.lstrip():
                    continue  # content change — covered by other rules
                orig_indent = len(orig) - len(orig.lstrip())
                exp_indent = len(expected) - len(expected.lstrip())
                if orig_indent == exp_indent:
                    continue
                line_no = i1 + di + 1
                if line_no in existing_nxf001:
                    continue
                fix = NXFEdit(
                    start_line=line_no,
                    start_col=0,
                    end_line=line_no,
                    end_col=orig_indent,
                    replacement=" " * exp_indent,
                )
                violations.append(
                    NXFViolation(
                        "NXF001",
                        line_no,
                        f"Wrong indentation level (found {orig_indent} spaces, expected {exp_indent})",
                        fix=fix,
                    )
                )
        return violations

    # ------------------------------------------------------------------
    # format() helpers
    # ------------------------------------------------------------------

    def _restore_closer_style(self, original: str, formatted: str) -> str:
        """Order-based mapping: preserve each closer's bare/explicit style from original."""
        orig_styles = [
            line.strip() == "__"
            for line in original.splitlines()
            if line.strip().startswith("__")
        ]
        if not any(orig_styles):
            return formatted
        result_lines: list[str] = []
        idx = 0
        for line in formatted.splitlines():
            if line.strip().startswith("__") and idx < len(orig_styles):
                if orig_styles[idx]:
                    ind = len(line) - len(line.lstrip())
                    result_lines.append(" " * ind + "__")
                    idx += 1
                    continue
                idx += 1
            result_lines.append(line)
        return "\n".join(result_lines)

    def _enforce_section_blank_lines(self, lines: list[str]) -> list[str]:
        """Ensure exactly 1 blank line between a section closer and the next section header."""
        result: list[str] = []
        for i, line in enumerate(lines):
            result.append(line)
            if is_closer(line) and indent_level(line) == 1:
                j = i + 1
                while j < len(lines) and lines[j].strip() == "":
                    j += 1
                if j < len(lines) and is_section_header(lines[j]):
                    blanks = j - (i + 1)
                    if blanks == 0:
                        result.append("")
        return self._collapse_multiple_blanks(result)

    def _collapse_multiple_blanks(self, lines: list[str]) -> list[str]:
        result: list[str] = []
        prev_blank = False
        for line in lines:
            is_blank = line.strip() == ""
            if is_blank and prev_blank:
                continue
            result.append(line)
            prev_blank = is_blank
        return result
