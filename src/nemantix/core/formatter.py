from __future__ import annotations

import math
import re
from dataclasses import dataclass

from nemantix.core.parser import ParserLark

_REDUNDANT_QUALIFIER_RE = re.compile(r"@completion:\s*(\w+)\s*->\s*(\w+)")
_MICROPROMPT_OPEN_RE = re.compile(r"(?<!>)>>(?![ >])")
_MICROPROMPT_CLOSE_RE = re.compile(r"(?<![ <])<<(?!<)")

_SECTION_KEYWORDS = frozenset({"body", "in", "out", "guidelines", "plan"})


@dataclass
class NXFViolation:
    rule: str
    line: int
    message: str


class Formatter:
    """Formatter for Nemantix source files.

    format() parses the source into an AST and re-serializes it, so any
    indentation error (wrong unit, missing levels, flat output) is corrected
    structurally.  check() runs text-level analysis and returns violations
    without modifying the source.

    strict=True enforces NXF101: all closers are emitted in their explicit form
    (__body, __action, etc.).  strict=False (default) preserves the closer style
    used in the original source via order-based mapping.
    """

    def __init__(self, strict: bool = True) -> None:
        self._strict = strict

    def format(self, source: str) -> str:
        stmts = ParserLark().parse(source, "<formatter>")
        serialized = "\n\n".join(stmt.to_nxs() for stmt in stmts)
        lines: list[str] = list(serialized.splitlines())
        lines = self._enforce_section_blank_lines(lines)
        result = "\n".join(lines)
        if not self._strict:
            result = self._restore_closer_style(source, result)
        return result

    def check(self, source: str) -> list[NXFViolation]:
        ParserLark().parse(source, "<formatter>")
        violations: list[NXFViolation] = []
        lines = source.splitlines()
        violations.extend(self._check_indentation(lines))
        violations.extend(self._check_line_lengths(lines))
        violations.extend(self._check_blank_lines(lines))
        violations.extend(self._check_annotation_style(lines))
        violations.extend(self._check_microprompt_spacing(lines))
        return violations

    # ------------------------------------------------------------------
    # NXF502 — redundant completion qualifier
    # ------------------------------------------------------------------

    def _check_annotation_style(self, lines: list[str]) -> list[NXFViolation]:
        violations: list[NXFViolation] = []
        for i, line in enumerate(lines, start=1):
            m = _REDUNDANT_QUALIFIER_RE.search(line)
            if m and m.group(1) == m.group(2):
                q = m.group(1)
                violations.append(
                    NXFViolation(
                        "NXF502",
                        i,
                        f"Redundant qualifier: use '@completion: {q}' instead of"
                        f" '@completion: {q}->{q}'",
                    )
                )
        return violations

    # ------------------------------------------------------------------
    # NXF202 — microprompt delimiter spacing
    # ------------------------------------------------------------------

    def _check_microprompt_spacing(self, lines: list[str]) -> list[NXFViolation]:
        violations: list[NXFViolation] = []
        for i, line in enumerate(lines, start=1):
            if line.lstrip().startswith("#"):
                continue
            if _MICROPROMPT_OPEN_RE.search(line) or _MICROPROMPT_CLOSE_RE.search(line):
                violations.append(
                    NXFViolation(
                        "NXF202",
                        i,
                        "Microprompt delimiters must be surrounded by one space: '>> content <<'",
                    )
                )
        return violations

    # ------------------------------------------------------------------
    # NXF001 — indentation (check only; format() delegates to to_nxs())
    # ------------------------------------------------------------------

    def _detect_indent_unit(self, lines: list[str]) -> int:
        sizes = set()
        for line in lines:
            stripped = line.lstrip(" ")
            leading = len(line) - len(stripped)
            if leading > 0:
                sizes.add(leading)
        if not sizes:
            return 0
        unit = sizes.pop()
        for s in sizes:
            unit = math.gcd(unit, s)
        return unit

    def _check_indentation(self, lines: list[str]) -> list[NXFViolation]:
        unit = self._detect_indent_unit(lines)
        if unit == 0 or unit == 2:
            return []
        violations = []
        for i, line in enumerate(lines, start=1):
            stripped = line.lstrip(" ")
            leading = len(line) - len(stripped)
            if leading > 0 and leading % 2 != 0:
                violations.append(
                    NXFViolation(
                        "NXF001",
                        i,
                        f"Indentation is not a multiple of 2 spaces (found {leading})",
                    )
                )
        if unit != 2:
            violations.append(
                NXFViolation(
                    "NXF001", 1, f"Indentation unit is {unit} spaces; expected 2"
                )
            )
        return violations

    # ------------------------------------------------------------------
    # NXF002 — line length
    # ------------------------------------------------------------------

    def _check_line_lengths(self, lines: list[str]) -> list[NXFViolation]:
        violations = []
        for i, line in enumerate(lines, start=1):
            if len(line) > 120:
                violations.append(
                    NXFViolation(
                        "NXF002", i, f"Line exceeds 120 characters ({len(line)})"
                    )
                )
        return violations

    # ------------------------------------------------------------------
    # NXF003, NXF004, NXF005 — blank lines
    # ------------------------------------------------------------------

    def _is_closer(self, line: str) -> bool:
        return line.lstrip().startswith("__")

    def _is_section_header(self, line: str) -> bool:
        stripped = line.strip()
        keyword = stripped.rstrip(":").strip()
        return stripped.endswith(":") and keyword in _SECTION_KEYWORDS

    def _indent_level(self, line: str, unit: int = 2) -> int:
        stripped = line.lstrip(" ")
        leading = len(line) - len(stripped)
        return leading // unit if unit else 0

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
                    indent = len(line) - len(line.lstrip())
                    result_lines.append(" " * indent + "__")
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
            if self._is_closer(line) and self._indent_level(line) == 1:
                j = i + 1
                while j < len(lines) and lines[j].strip() == "":
                    j += 1
                if j < len(lines) and self._is_section_header(lines[j]):
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

    def _check_blank_lines(self, lines: list[str]) -> list[NXFViolation]:
        violations: list[NXFViolation] = []

        # NXF005: blank line before a closer
        for i, line in enumerate(lines):
            if (
                line.strip() == ""
                and i + 1 < len(lines)
                and self._is_closer(lines[i + 1])
            ):
                violations.append(
                    NXFViolation("NXF005", i + 1, "Blank line before block closer")
                )

        # NXF003: top-level block separation
        for i, line in enumerate(lines):
            if line.strip() == "" or self._indent_level(line) != 0:
                continue
            if i > 0:
                prev = lines[i - 1]
                if (
                    prev.strip() != ""
                    and self._indent_level(prev) == 0
                    and self._is_closer(prev)
                ):
                    violations.append(
                        NXFViolation(
                            "NXF003",
                            i + 1,
                            "Missing blank line between top-level blocks",
                        )
                    )
            blank_count = 0
            j = i - 1
            while j >= 0 and lines[j].strip() == "":
                blank_count += 1
                j -= 1
            if blank_count > 1:
                violations.append(
                    NXFViolation(
                        "NXF003",
                        i + 1,
                        f"Multiple blank lines ({blank_count}) before top-level block",
                    )
                )

        # NXF004: section separator
        for i, line in enumerate(lines):
            if not (self._is_closer(line) and self._indent_level(line) == 1):
                continue
            j = i + 1
            while j < len(lines) and lines[j].strip() == "":
                j += 1
            if j < len(lines) and self._is_section_header(lines[j]):
                blanks = j - (i + 1)
                if blanks == 0:
                    violations.append(
                        NXFViolation(
                            "NXF004",
                            i + 1,
                            "Missing blank line between internal sections",
                        )
                    )

        return violations
