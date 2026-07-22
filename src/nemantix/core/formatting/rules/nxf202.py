from __future__ import annotations

import re

from nemantix.core.formatting._edit import NXFEdit
from nemantix.core.formatting._rule import NXFRule, NXFViolation
from nemantix.core.node import BlockStatement, FileMeta, MicroPrompt, Statement

_MICROPROMPT_OPEN_RE = re.compile(r"(?<!>)>>(?![ >])")
_MICROPROMPT_CLOSE_RE = re.compile(r"(?<![ <])<<(?!<)")


def _fix_spacing(line: str) -> str:
    line = re.sub(r"(?<!>)>>(?![ >])", ">> ", line)
    line = re.sub(r"(?<![ <])<<(?!<)", " <<", line)
    return line


def _collect_inline_microprompt_lines(stmts: list) -> set[int]:
    """Return 1-indexed source lines that contain an inline (non-block) MicroPrompt node.

    Block-prompt content (>>>...<<<) is captured as a single PROMPT_BLOCK_TEXT token by
    the lexer, so >>text<< inside a block never produces a separate MicroPrompt AST node
    and is never included in the returned set.
    """
    inline_lines: set[int] = set()
    stack: list = list(stmts)
    visited: set[int] = set()
    while stack:
        obj = stack.pop()
        oid = id(obj)
        if oid in visited:
            continue
        visited.add(oid)
        if isinstance(obj, MicroPrompt):
            if "\n" not in obj.prompt:
                fm = obj.meta.get("file_meta")
                if isinstance(fm, FileMeta):
                    inline_lines.add(fm.line[0])
            continue
        if isinstance(obj, Statement):
            if isinstance(obj, BlockStatement):
                stack.extend(obj.children)
            for attr, v in vars(obj).items():
                if attr in ("children", "meta"):
                    continue
                if isinstance(v, (MicroPrompt, Statement)):
                    stack.append(v)
                elif isinstance(v, list):
                    for item in v:
                        if isinstance(item, (MicroPrompt, Statement)):
                            stack.append(item)
    return inline_lines


class NXF202Rule(NXFRule):
    """Microprompt delimiters must be surrounded by one space: '>> content <<'."""

    code = "NXF202"

    def detect(self, stmts: list, lines: list[str]) -> list[NXFViolation]:
        violations: list[NXFViolation] = []
        inline_lines = _collect_inline_microprompt_lines(stmts)
        for i, line in enumerate(lines, start=1):
            if i not in inline_lines:
                continue
            if _MICROPROMPT_OPEN_RE.search(line) or _MICROPROMPT_CLOSE_RE.search(line):
                fixed_line = _fix_spacing(line)
                fix = NXFEdit(
                    start_line=i,
                    start_col=0,
                    end_line=i,
                    end_col=len(line),
                    replacement=fixed_line,
                )
                violations.append(
                    NXFViolation(
                        "NXF202",
                        i,
                        "Microprompt delimiters must be surrounded by one space: '>> content <<'",
                        fix=fix,
                    )
                )
        return violations
