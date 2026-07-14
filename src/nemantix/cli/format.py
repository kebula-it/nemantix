from __future__ import annotations

import argparse
import sys
from pathlib import Path

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
    p.add_argument("files", nargs="+", metavar="FILE", help="Nemantix source files to process")
    p.set_defaults(handler=handle_format)
    return p


def handle_format(args: argparse.Namespace) -> int:
    formatter = Formatter(strict=not args.permissive)
    overall_rc = 0

    for path_str in args.files:
        path = Path(path_str)
        if not path.exists():
            print(f"Error: {path_str}: file not found", file=sys.stderr)
            overall_rc = 1
            continue

        ext = path.suffix.lstrip(".").lower()
        script_type = extension_map.get(ext)
        if script_type is None:
            print(f"Error: {path_str}: unsupported extension (expected .nxs, .nxc, .nxv)", file=sys.stderr)
            overall_rc = 1
            continue

        source = path.read_text(encoding="utf-8")
        check_only = args.check or script_type is ScriptTypeEnum.NXV

        if check_only:
            violations = formatter.check(source)
            if violations:
                overall_rc = 1
                _print_violations(path_str, violations)
        else:
            formatted = formatter.format(source)
            if formatted != source:
                path.write_text(formatted, encoding="utf-8")

    return overall_rc


def _print_violations(path: str, violations: list[NXFViolation]) -> None:
    for v in violations:
        print(f"{path}:{v.line}: {v.rule} {v.message}")