from __future__ import annotations

from dataclasses import dataclass


@dataclass
class NXFEdit:
    """A targeted text replacement that fixes a single NXF violation.

    All positions are 1-indexed lines and 0-indexed columns, matching Lark's
    FileMeta convention.  end_col is exclusive.
    """

    start_line: int
    start_col: int
    end_line: int
    end_col: int
    replacement: str


def _to_offset(source: str, line: int, col: int) -> int:
    """Convert 1-indexed line + 0-indexed col to a character offset in *source*."""
    offset = 0
    for i, ln in enumerate(source.split("\n"), start=1):
        if i == line:
            return offset + col
        offset += len(ln) + 1  # +1 for the '\n'
    return len(source)


def apply_edits(source: str, edits: list[NXFEdit]) -> str:
    """Apply *edits* to *source* without mutual interference.

    All offsets are computed against the original source before any edit is
    applied, then edits are executed in descending order so that earlier
    positions are never shifted by later replacements.
    """
    if not edits:
        return source
    pairs = [
        (
            _to_offset(source, e.start_line, e.start_col),
            _to_offset(source, e.end_line, e.end_col),
            e.replacement,
        )
        for e in edits
    ]
    pairs.sort(key=lambda p: p[0], reverse=True)
    result = source
    for start, end, repl in pairs:
        result = result[:start] + repl + result[end:]
    return result
