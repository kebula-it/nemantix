from __future__ import annotations

import argparse
import sys
from pathlib import Path

from nemantix.core.script import Script
from nemantix.core.source_manager import LocalSourceManager
from nemantix.security.verifier import Verifier


def register(subparsers: argparse._SubParsersAction) -> argparse.ArgumentParser:  # type: ignore[type-arg]
    """Add the 'verify' subcommand to *subparsers* and return its parser."""
    p = subparsers.add_parser(
        "verify", help="Verify the cryptographic signature of NXV files"
    )
    p.add_argument("paths", nargs="*", help="NXV files to verify")
    p.add_argument("--key", required=True, help="Path to public key PEM")
    p.set_defaults(handler=handle)
    return p


def handle(args: argparse.Namespace) -> int:
    """Verify NXV file signatures. Returns an exit code."""
    if not args.paths:
        return 0

    verifier = Verifier(args.key)
    has_error = False

    for path in args.paths:
        script = Script(location=Path(path), source_manager=LocalSourceManager())
        try:
            ok = verifier.verify(script)
            if not ok:
                print(f"Verification failed for '{path}'", file=sys.stderr)
                has_error = True
        except Exception as exc:
            print(f"Error verifying '{path}': {exc}", file=sys.stderr)
            has_error = True

    return 1 if has_error else 0
