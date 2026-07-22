from __future__ import annotations

from nemantix.core.formatting._edit import NXFEdit
from nemantix.core.formatting._helpers import is_closer
from nemantix.core.formatting._rule import NXFRule, NXFViolation


class NXF005Rule(NXFRule):
    """Blank line immediately before a block closer.

    Section closers (__body, __in, __out, __plan) are not AST nodes, so detection is
    text-based: the grammar guarantees that any __-prefixed line is a closer.
    """

    code = "NXF005"

    def detect(self, stmts: list, lines: list[str]) -> list[NXFViolation]:
        violations: list[NXFViolation] = []
        i = 0
        while i < len(lines):
            if (
                lines[i].strip() != ""
                or i + 1 >= len(lines)
                or not is_closer(lines[i + 1])
            ):
                i += 1
                continue
            # Find the start of the blank run ending just before the closer.
            run_start = i
            while run_start > 0 and lines[run_start - 1].strip() == "":
                run_start -= 1
            fix = NXFEdit(
                start_line=run_start + 1,
                start_col=0,
                end_line=i + 2,
                end_col=0,
                replacement="",
            )
            violations.append(
                NXFViolation(
                    "NXF005",
                    run_start + 1,
                    "Blank line(s) before block closer",
                    fix=fix,
                )
            )
            i += 1
        return violations
