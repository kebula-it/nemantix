from __future__ import annotations

from nemantix.core.formatting._rule import NXFRule, NXFViolation


class NXF101Rule(NXFRule):
    """Generic closer '__' must be replaced with a specific closer (e.g. '__action', '__body')."""

    code = "NXF101"

    def detect(self, stmts: list, lines: list[str]) -> list[NXFViolation]:
        violations: list[NXFViolation] = []
        for i, line in enumerate(lines, start=1):
            if line.strip() == "__":
                violations.append(
                    NXFViolation(
                        "NXF101",
                        i,
                        "Generic closer '__' should use a specific form (e.g. '__action', '__body')",
                    )
                )
        return violations
