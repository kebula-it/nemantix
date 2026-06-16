from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

from nemantix.core.expertise import Expertise
from nemantix.security.verifier import DebugVerifier


def register(subparsers: argparse._SubParsersAction) -> argparse.ArgumentParser:  # type: ignore[type-arg]
    """Add the 'code' subcommand to *subparsers* and return its parser."""
    p = subparsers.add_parser(
        "code", help="Compile NXS scripts to NXC without executing them"
    )
    p.add_argument("paths", nargs="*", help="NXS scripts to compile")
    p.add_argument(
        "--output",
        dest="output",
        default=None,
        help="Output directory (default: same directory as source)",
    )
    p.add_argument(
        "--vendor",
        default=os.environ.get("NEMANTIX_VENDOR", "openai"),
        help="LLM vendor (env: NEMANTIX_VENDOR)",
    )
    p.add_argument(
        "--model",
        default=os.environ.get("NEMANTIX_MODEL", "gpt-5-mini"),
        help="LLM model name (env: NEMANTIX_MODEL)",
    )
    p.add_argument(
        "--credentials",
        default="credentials.json",
        help="Path to credentials JSON file",
    )
    p.set_defaults(handler=handle)
    return p


def handle(args: argparse.Namespace) -> int:
    """Compile NXS scripts to NXC. Returns an exit code."""
    if not args.paths:
        return 0

    try:
        expertise = Expertise.from_local_scripts(
            paths=[Path(p) for p in args.paths],
            verifier=DebugVerifier(),
            vendor=args.vendor,
            model=args.model,
            credentials_path=args.credentials,
            export_location=args.output,
        )
        expertise.build()
    except Exception as exc:
        print(f"Compilation error: {exc}", file=sys.stderr)
        return 1

    return 0
