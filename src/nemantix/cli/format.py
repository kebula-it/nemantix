from __future__ import annotations

import argparse
import sys
from pathlib import Path

from nemantix.core.exceptions import NemantixParserException
from nemantix.core.formatter import Formatter, NXFViolation
from nemantix.core.script import ScriptTypeEnum, extension_map


def register(subparsers: argparse._SubParsersAction) -> argparse.ArgumentParser:  # type: ignore[type-arg]
    """Add the 'format' subcommand to *subparsers* and return its parser."""
    p = subparsers.add_parser(
        "format",
        help="Format or check Nemantix source files",
        description=(
            "Format Nemantix source files in-place (default) or report formatting "
            "violations without modifying files (--check).  "
            ".nxv files are always check-only."
        ),
    )
    p.add_argument(
        "--check",
        action="store_true",
        default=False,
        help="Report violations and exit non-zero if any; do not modify files.",
    )
    p.add_argument(
        "--permissive",
        action="store_true",
        default=False,
        help="Preserve original closer style (bare __ vs __body, __action, etc.) instead of enforcing explicit form.",
    )
    p.add_argument(
        "files",
        nargs="+",
        metavar="FILE",
        help="Nemantix source files or directories to process",
    )
    p.set_defaults(handler=handle_format)
    return p


def _collect_paths(path_str: str) -> list[Path]:
    """Expand a file or directory to a list of formattable paths."""
    path = Path(path_str)
    if not path.exists():
        return []
    if path.is_dir():
        return sorted(
            p for ext in ("nxs", "nxc", "nxv") for p in path.rglob(f"*.{ext}")
        )
    return [path]


def handle_format(args: argparse.Namespace) -> int:
    formatter = Formatter(strict=not args.permissive)
    overall_rc = 0

    for path_str in args.files:
        if not Path(path_str).exists():
            print(f"Error: {path_str}: file not found", file=sys.stderr)
            overall_rc = 1
            continue

        for path in _collect_paths(path_str):
            overall_rc |= _process_file(formatter, path, args.check)

    return overall_rc


def _process_file(formatter: Formatter, path: Path, check: bool) -> int:
    """Process a single file. Returns 0 on success, 1 on any error or violation."""
    path_str = str(path)
    ext = path.suffix.lstrip(".").lower()
    script_type = extension_map.get(ext)
    if script_type is None:
        print(
            f"Error: {path_str}: unsupported extension (expected .nxs, .nxc, .nxv)",
            file=sys.stderr,
        )
        return 1

    try:
        source = path.read_text(encoding="utf-8")
    except UnicodeDecodeError:
        print(f"Error: {path_str}: file is not valid UTF-8", file=sys.stderr)
        return 1

    check_only = check or script_type is ScriptTypeEnum.NXV

    try:
        if check_only:
            violations = formatter.check(source)
            if violations:
                _print_violations(path_str, violations)
                return 1
        else:
            formatted = formatter.format(source)
            if formatted != source:
                try:
                    path.write_text(formatted, encoding="utf-8")
                except PermissionError:
                    print(f"Error: {path_str}: permission denied", file=sys.stderr)
                    return 1
    except (SyntaxError, NemantixParserException) as exc:
        print(f"Error: {path_str}: {exc}", file=sys.stderr)
        return 1

    return 0


def _print_violations(path: str, violations: list[NXFViolation]) -> None:
    for v in violations:
        print(f"{path}:{v.line}: {v.rule} {v.message}")
