from __future__ import annotations

from nemantix.core.node import BlockStatement, DoStatement

_SECTION_KEYWORDS = frozenset({"body", "in", "out", "guidelines", "plan"})


def collect_do_statements(stmts: list) -> list[DoStatement]:
    result: list[DoStatement] = []
    stack: list = list(stmts)
    while stack:
        obj = stack.pop()
        if isinstance(obj, DoStatement):
            result.append(obj)
        elif isinstance(obj, BlockStatement):
            stack.extend(obj.children)
    return result


def leading_indent(line: str) -> str:
    return line[: len(line) - len(line.lstrip())]


def is_closer(line: str) -> bool:
    return line.lstrip().startswith("__")


def is_section_header(line: str) -> bool:
    stripped = line.strip()
    keyword = stripped.rstrip(":").strip()
    return stripped.endswith(":") and keyword in _SECTION_KEYWORDS


def indent_level(line: str, unit: int = 2) -> int:
    stripped = line.lstrip(" ")
    leading = len(line) - len(stripped)
    return leading // unit if unit else 0
