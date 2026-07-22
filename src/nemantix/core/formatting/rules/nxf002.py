from __future__ import annotations

import re
import textwrap

from nemantix.core.formatting._edit import NXFEdit
from nemantix.core.formatting._helpers import collect_do_statements, leading_indent
from nemantix.core.formatting._rule import NXFRule, NXFViolation
from nemantix.core.node import FileMeta, ImportToolsetStatement

_MAX = 120
_INLINE_PROMPT_RE = re.compile(r">>\s*(.*?)\s*<<")


class NXF002Rule(NXFRule):
    code = "NXF002"

    def detect(self, stmts: list, lines: list[str]) -> list[NXFViolation]:
        # Lines covered by more specific rules (NXF401/402) are skipped entirely.
        covered: set[int] = set()
        for stmt in collect_do_statements(stmts):
            fm = stmt.meta.get("file_meta")
            if isinstance(fm, FileMeta) and fm.line[0] == fm.line[1]:
                covered.add(fm.line[0])
        for stmt in stmts:
            if isinstance(stmt, ImportToolsetStatement):
                fm = stmt.meta.get("file_meta")
                if isinstance(fm, FileMeta) and fm.line[0] == fm.line[1]:
                    covered.add(fm.line[0])

        violations: list[NXFViolation] = []
        for i, line in enumerate(lines):
            line_no = i + 1
            if len(line) <= _MAX or line_no in covered:
                continue
            fix = None
            stripped = line.strip()
            if stripped.startswith(">>") and not stripped.startswith(">>>"):
                m = _INLINE_PROMPT_RE.search(line)
                if m:
                    content = m.group(1)
                    indent = leading_indent(line)
                    wrapped = textwrap.wrap(content, _MAX)
                    replacement = (
                        indent + ">>>\n" + "\n".join(wrapped) + "\n" + indent + "<<<"
                    )
                    fix = NXFEdit(
                        start_line=line_no,
                        start_col=0,
                        end_line=line_no,
                        end_col=len(line),
                        replacement=replacement,
                    )
            violations.append(
                NXFViolation(
                    "NXF002",
                    line_no,
                    f"Line exceeds 120 characters ({len(line)})",
                    fix=fix,
                )
            )
        return violations
