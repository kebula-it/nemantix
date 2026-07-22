from __future__ import annotations

from nemantix.core.formatting._edit import NXFEdit
from nemantix.core.formatting._helpers import (
    collect_do_statements,
    is_closer,
    leading_indent,
)
from nemantix.core.formatting._rule import NXFRule, NXFViolation
from nemantix.core.node import FileMeta, MicroPrompt

_MAX = 120


class NXF402Rule(NXFRule):
    """'do' form: prefer inline when ≤ 120 chars, block when > 120 chars."""

    code = "NXF402"

    def detect(self, stmts: list, lines: list[str]) -> list[NXFViolation]:
        violations: list[NXFViolation] = []
        for stmt in collect_do_statements(stmts):
            fm = stmt.meta.get("file_meta")
            if not isinstance(fm, FileMeta):
                continue
            source_line = lines[fm.line[0] - 1]
            indent = leading_indent(source_line)

            # Build inline representation (needed for both directions)
            parts: list[str] = ["do"]
            if stmt.callable_type is not None:
                parts.append(str(stmt.callable_type.value).lower())
            parts.append(str(stmt.name))
            if stmt.using is not None:
                parts.append(f"using [{stmt.using.to_nxs()}]")
            if stmt.producing is not None:
                parts.append(f"producing [{stmt.producing.to_nxs()}]")
            if isinstance(stmt.producing_schema, str):
                parts.append(f"as {{{stmt.producing_schema}}}")
            if isinstance(stmt.prompt, MicroPrompt):
                parts.append(f">> {stmt.prompt.prompt} <<")
            inline = indent + " ".join(parts)

            if fm.line[0] >= fm.line[1]:
                # Inline form — check if it exceeds 120 chars
                if len(source_line) <= _MAX:
                    continue
                n = 3 if stmt.callable_type is not None else 2
                block_lines: list[str] = [indent + " ".join(parts[:n]) + ":"]
                if stmt.using is not None:
                    block_lines.append(indent + "  using [" + stmt.using.to_nxs() + "]")
                if stmt.producing is not None:
                    block_lines.append(
                        indent + "  producing [" + stmt.producing.to_nxs() + "]"
                    )
                if isinstance(stmt.producing_schema, str):
                    block_lines.append(indent + "  as {" + stmt.producing_schema + "}")
                closer = indent + "__do"
                if isinstance(stmt.prompt, MicroPrompt):
                    closer += f" >> {stmt.prompt.prompt} <<"
                block_lines.append(closer)
                fix = NXFEdit(
                    start_line=fm.line[0],
                    start_col=0,
                    end_line=fm.line[0],
                    end_col=len(source_line),
                    replacement="\n".join(block_lines),
                )
                violations.append(
                    NXFViolation(
                        "NXF402",
                        fm.line[0],
                        "Inline 'do' exceeds 120 chars; use block form",
                        fix=fix,
                    )
                )
            else:
                # Block form — check if inline would fit
                if isinstance(stmt.prompt, MicroPrompt) and "\n" in stmt.prompt.prompt:
                    continue
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
                        "NXF402",
                        fm.line[0],
                        "Block 'do' fits within 120 chars; use inline form",
                        fix=fix,
                    )
                )
        return violations
