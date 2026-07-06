from __future__ import annotations

import argparse
import sys
from importlib.metadata import EntryPoint

from nemantix.cli._normalize import _nemantix_entry_points, _normalize_argv

_FIRST_PARTY_DIST_NAME = "nemantix"


def build_parser() -> argparse.ArgumentParser:
    """Build the top-level argparse parser, discovering subcommands via entry points.

    First-party entry points (dist name "nemantix") are loaded first; entry points from
    other distributions are loaded last and win on name collision, letting
    third-party packages extend or override core subcommands.
    """
    p = argparse.ArgumentParser(
        prog="nemantix", description="Nemantix agentic AI runner"
    )
    subs = p.add_subparsers(dest="subcommand")

    seen: dict[str, EntryPoint] = {}
    for ep in sorted(
        _nemantix_entry_points(),
        key=lambda e: getattr(e.dist, "name", "") != _FIRST_PARTY_DIST_NAME,
    ):
        seen[ep.name] = ep  # last (post-sort) entry point wins on name collision

    for ep in seen.values():
        ep.load()(subs)

    return p


def main(argv: list[str] | None = None) -> int:
    """CLI entry point for the nemantix command."""
    raw = list(argv) if argv is not None else sys.argv[1:]
    p = build_parser()

    args = p.parse_args(_normalize_argv(raw))

    if not hasattr(args, "handler"):
        p.print_help()
        return 1

    return args.handler(args)
