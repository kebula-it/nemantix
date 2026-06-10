from __future__ import annotations

import argparse
import sys
from pathlib import Path

from nemantix.core.script import Script
from nemantix.core.source_manager import LocalSourceManager
from nemantix.security.signer import Signer


def register(subparsers: argparse._SubParsersAction) -> argparse.ArgumentParser:  # type: ignore[type-arg]
    """Add the 'sign' subcommand to *subparsers* and return its parser."""
    p = subparsers.add_parser("sign", help="Sign NXC files to produce NXV (verifiable)")
    p.add_argument("paths", nargs="*", help="NXC files to sign")
    p.add_argument("--key", required=True, help="Path to private key PEM")
    p.add_argument(
        "--output",
        dest="output",
        default=None,
        help="Output directory (default: same directory as source)",
    )
    p.set_defaults(handler=handle)
    return p


def handle(args: argparse.Namespace) -> int:
    """Sign NXC files. Returns an exit code."""
    if not args.paths:
        return 0

    signer = Signer(args.key)
    has_error = False

    for path in args.paths:
        script = Script(location=Path(path), source_manager=LocalSourceManager())
        try:
            signer.sign(script)
        except Exception as exc:
            print(f"Error signing '{path}': {exc}", file=sys.stderr)
            has_error = True

    return 1 if has_error else 0
