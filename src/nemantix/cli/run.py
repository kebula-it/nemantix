from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from nemantix.knowledge_base.core.nemantix_knowledge_base import KnowledgeBaseConfig

from nemantix.core.agent import Agent
from nemantix.core.expertise import Expertise
from nemantix.core.tools import Toolset
from nemantix.security.verifier import DebugVerifier, Verifier


def _build_kb_config(args: argparse.Namespace) -> "KnowledgeBaseConfig | None":
    if not args.use_knowledge_base:
        return None
    if not args.kb_view_ids:
        raise ValueError("--kb-view-ids is required when --use-knowledge-base is set.")
    username = os.environ.get("NEMANTIX_KB_USERNAME", "")
    if not username:
        raise ValueError(
            "NEMANTIX_KB_USERNAME environment variable is required when --use-knowledge-base is set."
        )
    password = os.environ.get("NEMANTIX_KB_PASSWORD", "")
    if not password:
        raise ValueError(
            "NEMANTIX_KB_PASSWORD environment variable is required when --use-knowledge-base is set."
        )
    from nemantix.knowledge_base.core.nemantix_knowledge_base import KnowledgeBaseConfig

    return KnowledgeBaseConfig(
        view_ids=args.kb_view_ids,
        db_engine=args.kb_db_engine,
        db_username=username,
        db_password=password,
        db_host=args.kb_db_host,
        db_port=args.kb_db_port,
        db_database=args.kb_db_database,
        base_storage_path=args.kb_base_storage_path,
        vector_subdir=args.kb_vector_subdir,
        vector_store_type=args.kb_vector_store_type,
    )


def _register_cli_toolsets(entries: list[str]) -> None:
    for entry in entries:
        if "=" in entry:
            cls_name, import_path = entry.split("=", 1)
            Toolset.register(import_path.strip(), cls_name.strip())
        else:
            Toolset.register(entry.strip())


def register(subparsers: argparse._SubParsersAction) -> argparse.ArgumentParser:
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
        default="coding_output",
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
    p.add_argument(
        "--toolset",
        action="append",
        default=[],
        metavar="TOOLSET",
        help=(
            "Toolset to load. Use 'ClassName=module.path' for a direct mapping "
            "or 'module.path' to add a package to the fallback lookup list. "
            "Repeatable."
        ),
    )
    # Knowledge Base flags
    p.add_argument(
        "--kb-view-id",
        dest="kb_view_ids",
        action="append",
        default=None,
        metavar="VIEW_ID",
        help="Knowledge Base view ID. Repeatable. Required with --use-knowledge-base.",
    )
    p.add_argument(
        "--kb-db-engine",
        dest="kb_db_engine",
        default="postgresql",
        help="KB database engine (default: postgresql)",
    )
    p.add_argument(
        "--kb-db-host",
        dest="kb_db_host",
        default="localhost",
        help="KB database host (default: localhost)",
    )
    p.add_argument(
        "--kb-db-port",
        dest="kb_db_port",
        type=int,
        default=5432,
        help="KB database port (default: 5432)",
    )
    p.add_argument(
        "--kb-db-database",
        dest="kb_db_database",
        default="nemantix_db",
        help="KB database name (default: nemantix_db)",
    )
    p.add_argument(
        "--kb-base-storage-path",
        dest="kb_base_storage_path",
        default="kb_storage",
        help="KB base storage path (default: kb_storage)",
    )
    p.add_argument(
        "--kb-vector-subdir",
        dest="kb_vector_subdir",
        default="vector_db",
        help="KB vector store subdirectory (default: vector_db)",
    )
    p.add_argument(
        "--kb-vector-store-type",
        dest="kb_vector_store_type",
        default="qdrant",
        help="KB vector store type (default: qdrant)",
    )
    p.set_defaults(handler=handle, _run_subparser=p)
    return p


def handle(args: argparse.Namespace) -> int:
    """Execute the 'run' subcommand. Returns an exit code."""
    if not args.paths:
        if hasattr(args, "_run_subparser"):
            args._run_subparser.print_help()
        return 0

    _register_cli_toolsets(args.toolset)
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

    try:
        kb_config = _build_kb_config(args)
    except ValueError as exc:
        print(f"Error: {exc}", file=sys.stderr)
        return 1

    agent = Agent(
        expertise=expertise,
        build_on_start=not args.no_build,
        use_embedder=args.use_embedder,
        use_knowledge_base=args.use_knowledge_base,
        kb_config=kb_config,
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
