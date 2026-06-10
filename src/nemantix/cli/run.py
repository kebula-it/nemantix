from __future__ import annotations

import argparse
import os
from pathlib import Path
from typing import Any

from nemantix.core.agent import Agent
from nemantix.core.expertise import Expertise
from nemantix.security.verifier import DebugVerifier, Verifier


def register(subparsers: argparse._SubParsersAction) -> argparse.ArgumentParser:  # type: ignore[type-arg]
    """Add the 'run' subcommand to *subparsers* and return its parser."""
    p = subparsers.add_parser("run", help="Execute NXS/NXC/NXV scripts")
    p.add_argument("paths", nargs="*", help="Scripts to execute")
    p.add_argument(
        "-u",
        "--user-request",
        dest="user_request",
        default=None,
        help="User request string (if omitted, read from stdin)",
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
    p.add_argument(
        "--export-location",
        dest="export_location",
        default=None,
        help="Directory for compiled script export",
    )
    p.add_argument(
        "--no-build",
        dest="no_build",
        action="store_true",
        help="Disable build-on-start",
    )
    p.add_argument(
        "--use-embedder",
        dest="use_embedder",
        action="store_true",
        help="Enable the sentence-transformer embedder",
    )
    p.add_argument(
        "--use-knowledge-base",
        dest="use_knowledge_base",
        action="store_true",
        help="Enable the knowledge base",
    )
    p.add_argument(
        "--log-level",
        dest="log_level",
        choices=["DEBUG", "INFO", "WARNING", "ERROR"],
        default=None,
        help="Agent logger level",
    )
    p.add_argument(
        "--verify",
        default=None,
        help="Path to public key PEM for NXV signature verification",
    )
    p.add_argument(
        "--debug", action="store_true", help="Enable the interactive CLI debugger (ndb)"
    )
    p.add_argument(
        "--profile",
        action="store_true",
        help="Enable the Profiler and print stats on exit",
    )
    p.set_defaults(handler=handle, _run_subparser=p)
    return p


def handle(args: argparse.Namespace) -> int:
    """Execute the 'run' subcommand. Returns an exit code."""
    if not args.paths:
        if hasattr(args, "_run_subparser"):
            args._run_subparser.print_help()
        return 0

    verifier = Verifier(args.verify) if args.verify else DebugVerifier()

    observers: list[Any] = []
    if args.debug:
        from nemantix.hub import Debugger

        observers.append(Debugger())
    if args.profile:
        from nemantix.hub import Profiler

        observers.append(Profiler())

    expertise = Expertise.from_local_scripts(
        paths=[Path(p) for p in args.paths],
        verifier=verifier,
        vendor=args.vendor,
        model=args.model,
        credentials_path=args.credentials,
        export_location=args.export_location,
        observers=observers or None,
    )

    agent = Agent(
        expertise=expertise,
        build_on_start=not args.no_build,
        use_embedder=args.use_embedder,
        use_knowledge_base=args.use_knowledge_base,
        log_level=args.log_level,
    )

    user_request: str = args.user_request or input("User request: ")
    exc, output = agent.run(user_request)

    if exc is not None:
        return 1

    if output is not None:
        print(output)

    if args.profile:
        for obs in observers:
            if hasattr(obs, "print_stats"):
                obs.print_stats()

    return 0
