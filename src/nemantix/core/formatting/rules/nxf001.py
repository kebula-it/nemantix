from __future__ import annotations

import math

from nemantix.core.formatting._edit import NXFEdit
from nemantix.core.formatting._rule import NXFRule, NXFViolation


def _detect_indent_unit(lines: list[str]) -> int:
    sizes: set[int] = set()
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


class NXF001Rule(NXFRule):
    code = "NXF001"

    def detect(self, stmts: list, lines: list[str]) -> list[NXFViolation]:
        unit = _detect_indent_unit(lines)
        if unit == 0 or unit == 2:
            return []
        violations: list[NXFViolation] = []
        for i, line in enumerate(lines, start=1):
            stripped = line.lstrip(" ")
            leading = len(line) - len(stripped)
            if leading > 0 and leading % unit != 0:
                violations.append(
                    NXFViolation(
                        "NXF001",
                        i,
                        f"Indentation is not a multiple of {unit} spaces (found {leading})",
                    )
                )
        if unit != 2:
            reindented = []
            for ln in lines:
                stripped = ln.lstrip(" ")
                leading = len(ln) - len(stripped)
                level = leading // unit
                reindented.append(" " * (level * 2) + stripped)
            fix = NXFEdit(
                start_line=1,
                start_col=0,
                end_line=len(lines),
                end_col=len(lines[-1]),
                replacement="\n".join(reindented),
            )
            violations.append(
                NXFViolation(
                    "NXF001",
                    1,
                    f"Indentation unit is {unit} spaces; expected 2",
                    fix=fix,
                )
            )
        return violations
