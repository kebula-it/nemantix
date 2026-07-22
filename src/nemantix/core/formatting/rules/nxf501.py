from __future__ import annotations

import re

from nemantix.core.formatting._edit import NXFEdit
from nemantix.core.formatting._rule import NXFRule, NXFViolation

# Matches an annotation (@key: value) or a label ({name}) at the start of a line.
_ANNOTATION_LINE_RE = re.compile(r"^\s*(@[\w.]+\s*:|{[^}]+})")


class NXF501Rule(NXFRule):
    """Annotation or label not indented at the same level as the block it introduces."""

    code = "NXF501"

    def detect(self, stmts: list, lines: list[str]) -> list[NXFViolation]:
        violations: list[NXFViolation] = []
        for i, line in enumerate(lines):
            if not _ANNOTATION_LINE_RE.match(line):
                continue
            annotation_indent = len(line) - len(line.lstrip())
            # Find the next non-annotation, non-blank line — that is the block header.
            j = i + 1
            while j < len(lines) and (
                lines[j].strip() == "" or _ANNOTATION_LINE_RE.match(lines[j])
            ):
                j += 1
            if j < len(lines):
                block_indent = len(lines[j]) - len(lines[j].lstrip())
                if annotation_indent != block_indent:
                    fixed_line = " " * block_indent + line.lstrip()
                    fix = NXFEdit(
                        start_line=i + 1,
                        start_col=0,
                        end_line=i + 1,
                        end_col=len(line),
                        replacement=fixed_line,
                    )
                    violations.append(
                        NXFViolation(
                            "NXF501",
                            i + 1,
                            f"Annotation at indent {annotation_indent} does not match"
                            f" block header at indent {block_indent}",
                            fix=fix,
                        )
                    )
        return violations
