from __future__ import annotations

from nemantix.core.formatting._edit import NXFEdit
from nemantix.core.formatting._helpers import indent_level
from nemantix.core.formatting._rule import NXFRule, NXFViolation
from nemantix.core.node import ActionBlock, Deliberate, FileMeta, PythonToolDeclaration


def _build_significant_line_sets(stmts: list) -> tuple[set[int], set[int]]:
    """Return (sig_starts, sig_ends): 1-indexed lines where significant blocks start/end.

    Significant blocks are ActionBlock, Deliberate, and PythonToolDeclaration — the only
    top-level constructs that require a blank line both before and after.  Import-type
    statements (Require, ImportToolsetStatement) are intentionally excluded so that
    consecutive imports can appear without blank lines between them.
    """
    sig_starts: set[int] = set()
    sig_ends: set[int] = set()
    for stmt in stmts:
        if isinstance(stmt, (ActionBlock, Deliberate, PythonToolDeclaration)):
            fm = stmt.meta.get("file_meta")
            if isinstance(fm, FileMeta):
                sig_starts.add(fm.line[0])
                sig_ends.add(fm.line[1])
    return sig_starts, sig_ends


class NXF003Rule(NXFRule):
    """Missing or excess blank lines between top-level blocks."""

    code = "NXF003"

    def detect(self, stmts: list, lines: list[str]) -> list[NXFViolation]:
        violations: list[NXFViolation] = []
        sig_starts, sig_ends = _build_significant_line_sets(stmts)
        for i, line in enumerate(lines):
            if line.strip() == "" or indent_level(line) != 0:
                continue
            if i > 0:
                prev = lines[i - 1]
                if (
                    prev.strip() != ""
                    and indent_level(prev) == 0
                    and not prev.lstrip().startswith(("@", "#", "{"))
                    and (i + 1 in sig_starts or i in sig_ends)
                ):
                    violations.append(
                        NXFViolation(
                            "NXF003",
                            i + 1,
                            "Missing blank line between top-level blocks",
                            fix=NXFEdit(
                                start_line=i + 1,
                                start_col=0,
                                end_line=i + 1,
                                end_col=0,
                                replacement="\n",
                            ),
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
                        fix=NXFEdit(
                            start_line=j + 3,
                            start_col=0,
                            end_line=i + 1,
                            end_col=0,
                            replacement="",
                        ),
                    )
                )
            elif blank_count == 1 and j < 0:
                violations.append(
                    NXFViolation(
                        "NXF003",
                        i + 1,
                        "Leading blank line before first top-level block",
                        fix=NXFEdit(
                            start_line=1,
                            start_col=0,
                            end_line=i + 1,
                            end_col=0,
                            replacement="",
                        ),
                    )
                )
        return violations
