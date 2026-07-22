from __future__ import annotations

from nemantix.core.formatting._edit import NXFEdit
from nemantix.core.formatting._helpers import is_closer, leading_indent
from nemantix.core.formatting._rule import NXFRule, NXFViolation
from nemantix.core.node import FileMeta, ImportToolsetStatement

_MAX = 120


class NXF401Rule(NXFRule):
    """Import form: prefer inline when ≤ 120 chars, block when > 120 chars."""

    code = "NXF401"

    def detect(self, stmts: list, lines: list[str]) -> list[NXFViolation]:
        violations: list[NXFViolation] = []
        for stmt in stmts:
            if not isinstance(stmt, ImportToolsetStatement):
                continue
            fm = stmt.meta.get("file_meta")
            if not isinstance(fm, FileMeta):
                continue
            source_line = lines[fm.line[0] - 1]
            indent = leading_indent(source_line)

            if fm.line[0] >= fm.line[1]:
                # Inline form — check if it exceeds 120 chars
                if len(source_line) <= _MAX:
                    continue
                header = f"{indent}from toolset {stmt.name}"
                if stmt.alias is not None:
                    header += f" as {stmt.alias}"
                if stmt.args is not None:
                    header += f" with [{stmt.args.to_nxs()}]"
                header += " use:"
                tools_line = indent + "  " + ", ".join(stmt.elements)
                block_str = header + "\n" + tools_line + "\n" + indent + "__use"
                fix = NXFEdit(
                    start_line=fm.line[0],
                    start_col=0,
                    end_line=fm.line[0],
                    end_col=len(source_line),
                    replacement=block_str,
                )
                violations.append(
                    NXFViolation(
                        "NXF401",
                        fm.line[0],
                        "Inline import exceeds 120 chars; use block form",
                        fix=fix,
                    )
                )
            else:
                # Block form — check if inline would fit
                inline = indent + stmt.to_nxs()
                if len(inline) > _MAX:
                    continue
                closer_line = min(fm.line[1] - 1, len(lines) - 1)
                while closer_line < len(lines) and not is_closer(lines[closer_line]):
                    closer_line += 1
                if closer_line >= len(lines):
                    continue
                fix = NXFEdit(
                    start_line=fm.line[0],
                    start_col=0,
                    end_line=closer_line + 1,
                    end_col=len(lines[closer_line]),
                    replacement=inline,
                )
                violations.append(
                    NXFViolation(
                        "NXF401",
                        fm.line[0],
                        "Block import fits within 120 chars; use inline form",
                        fix=fix,
                    )
                )
        return violations
