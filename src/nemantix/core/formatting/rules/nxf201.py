from __future__ import annotations

import re

from nemantix.core.formatting._edit import NXFEdit
from nemantix.core.formatting._rule import NXFRule, NXFViolation

# Matches a complete >>>...<<< block prompt on a single source line (non-greedy).
_SINGLE_LINE_BLOCK_PROMPT_RE = re.compile(r">>>(.*?)<<<")


class NXF201Rule(NXFRule):
    """Block prompt used when inline form would fit within 120 characters."""

    code = "NXF201"

    def detect(self, stmts: list, lines: list[str]) -> list[NXFViolation]:
        violations: list[NXFViolation] = []
        for i, line in enumerate(lines, start=1):
            m = _SINGLE_LINE_BLOCK_PROMPT_RE.search(line)
            if not m:
                continue
            # Replace >>>text<<< with >> text << (NXF202 requires spaces around content)
            content = m.group(1).strip()
            inline_line = line[: m.start()] + f">> {content} <<" + line[m.end() :]
            if len(inline_line) <= 120:
                fix = NXFEdit(
                    start_line=i,
                    start_col=0,
                    end_line=i,
                    end_col=len(line),
                    replacement=inline_line,
                )
                violations.append(
                    NXFViolation(
                        "NXF201",
                        i,
                        "Block prompt fits within 120 chars; use inline form '>> text <<'",
                        fix=fix,
                    )
                )
        return violations
