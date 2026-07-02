from __future__ import annotations

import argparse
import sys
from pathlib import Path

from nemantix.security.ecdsa import generate_keys


def register(subparsers: argparse._SubParsersAction) -> argparse.ArgumentParser:  # type: ignore[type-arg]
    """Add the 'keygen' subcommand to *subparsers* and return its parser."""
    p = subparsers.add_parser(
        "keygen", help="Generate an ECDSA key pair for signing and verification"
    )
    p.add_argument(
        "--output",
        default=".",
        help="Directory where the key files will be written (default: current directory)",
    )
    p.set_defaults(handler=handle)
    return p


def handle(args: argparse.Namespace) -> int:
    """Generate ECDSA key pair. Returns an exit code."""
    output = Path(args.output)
    if not output.is_dir():
        print(f"Error: output directory '{output}' does not exist.", file=sys.stderr)
        return 1
    try:
        generate_keys(output)
    except Exception as exc:
        print(f"Error generating keys: {exc}", file=sys.stderr)
        return 1
    print(f"Keys generated in '{output}':")
    print(f"  Private key: {output / 'nmx_ecdsa_private.pem'}")
    print(f"  Public key:  {output / 'nmx_ecdsa_public.pem'}")
    return 0
