from __future__ import annotations

import re

from nemantix.core.formatting._edit import NXFEdit
from nemantix.core.formatting._rule import NXFRule, NXFViolation

_REDUNDANT_QUALIFIER_RE = re.compile(r"@completion:\s*(\w+)\s*->\s*(\w+)")


class NXF502Rule(NXFRule):
    """Redundant completion qualifier: '@completion: x->x' should be '@completion: x'."""

    code = "NXF502"

    def detect(self, stmts: list, lines: list[str]) -> list[NXFViolation]:
        violations: list[NXFViolation] = []
        for i, line in enumerate(lines, start=1):
            m = _REDUNDANT_QUALIFIER_RE.search(line)
            if m and m.group(1) == m.group(2):
                q = m.group(1)
                fixed_line = line[: m.start()] + f"@completion: {q}" + line[m.end() :]
                fix = NXFEdit(
                    start_line=i,
                    start_col=0,
                    end_line=i,
                    end_col=len(line),
                    replacement=fixed_line,
                )
                violations.append(
                    NXFViolation(
                        "NXF502",
                        i,
                        f"Redundant qualifier: use '@completion: {q}' instead of"
                        f" '@completion: {q}->{q}'",
                        fix=fix,
                    )
                )
        return violations
