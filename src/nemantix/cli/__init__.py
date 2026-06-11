from __future__ import annotations

import argparse
import sys
from importlib.metadata import EntryPoint, entry_points

from nemantix.cli._normalize import _normalize_argv


def main(argv: list[str] | None = None) -> int:
    """CLI entry point for the nemantix command."""
    raw = list(argv) if argv is not None else sys.argv[1:]
    p = argparse.ArgumentParser(
        prog="nemantix", description="Nemantix agentic AI runner"
    )
    subs = p.add_subparsers(dest="subcommand")

    seen: dict[str, EntryPoint] = {}
    for ep in entry_points(group="nemantix"):
        seen[ep.name] = ep  # last installed entry point wins on name collision

    for ep in seen.values():
        ep.load()(subs)

    args = p.parse_args(_normalize_argv(raw))

    if not hasattr(args, "handler"):
        p.print_help()
        return 1

    return args.handler(args)
