from __future__ import annotations

from nemantix.core.formatting._edit import NXFEdit
from nemantix.core.formatting._helpers import indent_level, is_closer, is_section_header
from nemantix.core.formatting._rule import NXFRule, NXFViolation


class NXF004Rule(NXFRule):
    """Missing or excess blank lines between internal sections.

    Section closers (__body, __in, __out, __plan) and section headers (body:, in:, etc.)
    are not modelled as separate AST nodes — they are structural parts of ActionBlock /
    Deliberate.  Text-based detection is reliable because the grammar guarantees that
    __-prefixed lines are always closers and that section keywords only appear as headers.
    """

    code = "NXF004"

    def detect(self, stmts: list, lines: list[str]) -> list[NXFViolation]:
        violations: list[NXFViolation] = []
        for i, line in enumerate(lines):
            if not (is_closer(line) and indent_level(line) == 1):
                continue
            j = i + 1
            while j < len(lines) and lines[j].strip() == "":
                j += 1
            if j < len(lines) and is_section_header(lines[j]):
                blanks = j - (i + 1)
                if blanks == 0:
                    violations.append(
                        NXFViolation(
                            "NXF004",
                            i + 1,
                            "Missing blank line between internal sections",
                            fix=NXFEdit(
                                start_line=j + 1,
                                start_col=0,
                                end_line=j + 1,
                                end_col=0,
                                replacement="\n",
                            ),
                        )
                    )
                elif blanks > 1:
                    violations.append(
                        NXFViolation(
                            "NXF004",
                            i + 1,
                            f"Multiple blank lines ({blanks}) between internal sections",
                            fix=NXFEdit(
                                start_line=i + 3,
                                start_col=0,
                                end_line=j + 1,
                                end_col=0,
                                replacement="",
                            ),
                        )
                    )
        return violations
